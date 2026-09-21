from __future__ import annotations

import os
import re
import subprocess
from sys import argv

config = {
    "executable": r"C:\Program Files (x86)\DOSBox-0.74\DOSBox.exe",
    "relativeFileDirname": argv[1],
    "fileBasename": argv[2],
    "fileDirname": argv[3],
    "fileExt": argv[2].split('.')[-1],
    "fileBasenameNoExtension": ".".join(argv[2].split('.')[:-1]),
    "filePath": os.path.join(argv[3], argv[2])
}


def emulator():
    args = [
        "-noconsole",
        "-userconf",
        "-c",
        f"mount c {os.path.abspath(config['fileDirname'])}",
        "-c",
        "c:",
        "-c",
        f"cd {config['relativeFileDirname']}",
        "-c",
        f"tasm /zi {config['fileBasenameNoExtension']}.asm",
        "-c",
        f"tlink /v {config['fileBasenameNoExtension']}.obj",
        "-c",
        f"{config['fileBasenameNoExtension']}.exe",
        "-c",
        "pause",
        "-c",
        "exit"
    ]

    command = [config["executable"]] + args
    print(command)

    return subprocess.Popen(command)


class Parser:

    def __init__(self):
        self.procs: list[str] = []
        self.data: list[str] = []
        self.header: list[str] = []
        self.start: list[str] = []
        self.parsed_files = set()

    def parse_file(self, file_path: str) -> None:
        if file_path in self.parsed_files:
            return

        self.parsed_files.add(file_path)

        lines = self.read_file(file_path)

        self.parse_imports(file_path, lines)
        self.data.extend(self.parse_data(lines))
        self.procs.extend(self.parse_procedures(lines, file_path))
        self.header.extend(self.parse_header(lines))
        self.start.extend(self.parse_start(lines))

    def parse_imports(self, file_path: str, lines: list[str]) -> None:
        for line in lines:
            found_import = re.match(r"^\s*IMPORT\s+(.+)$", line, re.IGNORECASE)
            if not found_import:
                continue

            import_path = found_import.group(1).strip()
            base_directory = os.path.dirname(file_path)
            imported_file_path = os.path.join(base_directory, import_path)

            self.parse_file(os.path.abspath(imported_file_path))

    @staticmethod
    def parse_start(lines: list[str]) -> list[str]:
        start_line = -1
        end_line = -1

        for i, line in enumerate(lines):
            if re.match(r"^\s*start\s*:", line, re.IGNORECASE):
                start_line = i

            elif re.match(r"^\s*END\s+start\b", line, re.IGNORECASE):
                end_line = i
                break

        if start_line == -1:
            return []

        if end_line == -1:
            end_line = len(lines)

        return lines[start_line:end_line + 1]

    @staticmethod
    def parse_header(lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            if line.strip().upper() == 'DATASEG':
                return lines[:i]
        return []

    @staticmethod
    def parse_data(lines: list[str]) -> list[str]:
        data_start = Parser.find_data_start(lines)

        if data_start == -1:
            return []

        data = []
        for line in lines[data_start + 1:]:
            if line.strip().upper() == "CODESEG":
                break
            data.append(line)

        return data

    @staticmethod
    def find_data_start(lines: list[str]) -> int:
        for i, line in enumerate(lines):
            if line.strip().upper() == "DATASEG":
                return i

        return -1

    @staticmethod
    def parse_procedures(lines: list[str], file_path: str) -> list[str]:
        procs: list[str] = []

        current_line = 0
        while current_line < len(lines):
            proc_start = re.match(r"^\s*proc\s+(\S+)", lines[current_line], re.IGNORECASE)
            if not proc_start:
                current_line += 1
                continue

            proc_name = proc_start.group(1)
            starting_line = current_line
            procs.append(';' + file_path + '\n')  # for debug

            current_line += 1
            while current_line < len(lines):
                proc_end = re.match(r"^\s*endp\b", lines[current_line], re.IGNORECASE)
                if proc_end:
                    break

                current_line += 1

            proc_lines = lines[starting_line:current_line + 1]
            for l in proc_lines:
                procs.append(l.replace('.', proc_name))

        return procs

    @staticmethod
    def read_file(file_path: str) -> list[str]:
        with open(file_path, "r") as file:
            return file.readlines()

    def bake(self, file_path: str):
        lines: list[str] = []

        lines.extend(self.header)
        lines.append('DATASEG\n')
        lines.extend(self.data)
        lines.append('CODESEG\n')
        lines.extend(self.procs)
        lines.extend(self.start)

        lines = [line for line in lines
                 if not re.match(r'^\s*IMPORT\s+', line, re.IGNORECASE)]

        with open(file_path, 'w') as f:
            f.writelines(lines)


def main() -> None:
    match config["fileExt"]:
        case 'asm':
            emulator().wait()

        case 'noasm':
            file_path = os.path.abspath(config["filePath"])
            parser = Parser()
            parser.parse_file(file_path)
            parser.bake(file_path.replace('noasm', 'asm'))
            emulator().wait()

        case _:
            print(f"No action found for filetype "f"'{config['fileExt']}'")
            exit()

    base = os.path.join(config["relativeFileDirname"], config["fileBasenameNoExtension"].upper())

    os.remove(base + '.EXE')
    os.remove(base + '.MAP')
    os.remove(base + '.OBJ')


if __name__ == "__main__":
    main()
