DATASEG

procPtr dw 0, 0

CODESEG
;C:\asm\lib\utility.noasm
proc executeProcIfTrue
    push bp
    mov bp, sp

    cmp [bp + 4], 0
    je executeProcIfTrueend

    call [bp + 6]

executeProcIfTrueend:
    pop bp
    ret 4
endp executeProcIfTrue