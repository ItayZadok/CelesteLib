let documentation;

async function loadDocumentation() {
    const response = await fetch("docs.json");
    documentation = await response.json();

    createFileList();

    const firstFile = Object.keys(documentation.procedures)[0];
    showFile(firstFile);
}

function createFileList() {
    const container = document.getElementById("files");

    for (const file of Object.keys(documentation.procedures)) {
        const link = document.createElement("a");
        link.className = "file-link";
        link.textContent = file;
        link.onclick = () => showFile(file);
        container.appendChild(link);
    }
}

function showFile(file) {
    document.getElementById("file-title").textContent = file;

    const container = document.getElementById("procedures");
    container.innerHTML = "";

    for (const procedure of documentation.procedures[file]) {
        const section = document.createElement("div");
        section.className = "procedure";

        const name = document.createElement("h3");
        name.textContent = procedure.name;
        section.appendChild(name);

        const table = document.createElement("table");
        table.innerHTML =
            "<tr><th>Input</th><td>" + escapeHtml(procedure.inputs) + "</td></tr>" +
            "<tr><th>Output</th><td>" + escapeHtml(procedure.outputs) + "</td></tr>";
        section.appendChild(table);

        const description = document.createElement("p");
        description.className = "description";
        description.textContent = procedure.functionality;
        section.appendChild(description);

        container.appendChild(section);
    }
}

function escapeHtml(value) {
    const element = document.createElement("div");
    element.textContent = value;
    return element.innerHTML;
}

loadDocumentation();
