Unicode true
RequestExecutionLevel user
ManifestDPIAware true
SetCompressor /FINAL zlib

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"

!ifndef PAYLOAD_DIR
    !error "Supply /DPAYLOAD_DIR=<absolute portable application directory>"
!endif
!ifndef OUTPUT_EXE
    !error "Supply /DOUTPUT_EXE=<absolute Setup.exe path>"
!endif
!ifndef BUILD_PYTHON
    !define BUILD_PYTHON "python.exe"
!endif
!tempfile AVTR_FILE_COMMANDS
!system '"${BUILD_PYTHON}" "${__FILEDIR__}\installer_files.py" "${PAYLOAD_DIR}" "${AVTR_FILE_COMMANDS}"' = 0
!include "${AVTR_FILE_COMMANDS}"
!delfile "${AVTR_FILE_COMMANDS}"

!define APP_KEY "Software\RouterVisionStudio\AVTR"
!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\RouterVisionStudio-AVTR"
Name "AVTR Router Vision Studio"
OutFile "${OUTPUT_EXE}"
InstallDir "$LOCALAPPDATA\Programs\AVTR"
!ifndef AVTR_TEST_SHELL_DIR
    InstallDirRegKey HKCU "${APP_KEY}" "InstallDir"
!endif
ShowInstDetails show
ShowUninstDetails show
Var Portable

Function .onInit
    StrCpy $Portable 0
    ${GetParameters} $0
    ClearErrors
    ${GetOptions} $0 "/PORTABLE" $1
    ${IfNot} ${Errors}
        StrCpy $Portable 1
    ${EndIf}
    ClearErrors
FunctionEnd

!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\AVTR.exe"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "Thai"

!ifdef AVTR_TEST_SHELL_DIR
    !define SHORTCUT_DIR "${AVTR_TEST_SHELL_DIR}"
!else
    !define SHORTCUT_DIR "$SMPROGRAMS"
!endif

Section "AVTR"
    SetShellVarContext current
    ; Protect other software if the directory picker is pointed at an existing runtime.
    ReadINIStr $0 "$INSTDIR\.avtr-installer.ini" "AVTR" "Product"
    ${If} $0 != "RouterVisionStudio-AVTR"
        IfFileExists "$INSTDIR\AVTR.exe" occupied
        IfFileExists "$INSTDIR\_runtime\*.*" occupied
        IfFileExists "$INSTDIR\router_vision\*.*" occupied
        IfFileExists "$INSTDIR\production_app.py" occupied
        IfFileExists "$INSTDIR\Uninstall-AVTR.exe" occupied
    ${Else}
        ReadINIStr $1 "$INSTDIR\.avtr-installer.ini" "AVTR" "Portable"
        ${If} $1 != $Portable
            Goto occupied
        ${EndIf}
    ${EndIf}
    Goto install
    occupied:
        MessageBox MB_ICONSTOP|MB_OK "This folder contains another application. Choose an empty folder or the existing AVTR installation." /SD IDOK
        SetErrorLevel 2
        Abort
    install:
    !insertmacro InstallPortableFiles
    SetOutPath "$INSTDIR"
    WriteINIStr "$INSTDIR\.avtr-installer.ini" "AVTR" "Product" "RouterVisionStudio-AVTR"
    WriteINIStr "$INSTDIR\.avtr-installer.ini" "AVTR" "Portable" "$Portable"
    ${If} $Portable == 1
        Goto done
    ${EndIf}
    WriteUninstaller "$INSTDIR\Uninstall-AVTR.exe"
    CreateDirectory "${SHORTCUT_DIR}"
    CreateShortcut "${SHORTCUT_DIR}\AVTR Router Vision Studio.lnk" "$INSTDIR\AVTR.exe"
    !ifndef AVTR_TEST_SHELL_DIR
        WriteRegStr HKCU "${APP_KEY}" "InstallDir" "$INSTDIR"
        WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "AVTR Router Vision Studio"
        WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
        WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\AVTR.exe"
        WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\Uninstall-AVTR.exe"'
        WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
        WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
    !endif
    done:
SectionEnd

Section "Uninstall"
    SetShellVarContext current
    ReadINIStr $0 "$INSTDIR\.avtr-installer.ini" "AVTR" "Product"
    ${If} $0 != "RouterVisionStudio-AVTR"
        MessageBox MB_ICONSTOP|MB_OK "AVTR installation record is missing." /SD IDOK
        SetErrorLevel 2
        Abort
    ${EndIf}
    !insertmacro UninstallPortableFiles
    !ifdef AVTR_TEST_SHELL_DIR
        Delete "${SHORTCUT_DIR}\AVTR Router Vision Studio.lnk"
    !else
        ReadRegStr $0 HKCU "${APP_KEY}" "InstallDir"
        ${If} $0 == $INSTDIR
            Delete "${SHORTCUT_DIR}\AVTR Router Vision Studio.lnk"
            DeleteRegKey HKCU "${UNINSTALL_KEY}"
            DeleteRegKey HKCU "${APP_KEY}"
        ${EndIf}
    !endif
    Delete "$INSTDIR\.avtr-installer.ini"
    Delete "$INSTDIR\Uninstall-AVTR.exe"
    RMDir "$INSTDIR"
SectionEnd
