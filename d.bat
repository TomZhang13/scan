REM delete all output files, simply enter "d" in the command line

@echo off
del /q ".\figures\*.*"
del /q ".\layouts\*.*"
del /q ".\tables\*.*"
del /q ".\out_json\*.*"
echo Files deleted from figures and layouts.
