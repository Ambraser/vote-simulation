@echo off
setlocal EnableDelayedExpansion

:: Use ASCII-safe title to avoid code page issues
title vote_simulation UI

echo ============================================================
echo  vote_simulation UI - Lanceur Windows
echo ============================================================
echo.

:: ── Se placer dans le dossier du script ─────────────────────────────────────
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
if errorlevel 1 (
    echo [ERREUR] Impossible de naviguer vers le dossier du script : %SCRIPT_DIR%
    goto :error_pause
)

:: ============================================================
:: 1. Verifier / installer uv
:: ============================================================
where uv >nul 2>&1
if not errorlevel 1 goto :uv_ok

echo [INFO] uv non trouve. Installation via PowerShell...
echo.

:: Installer uv
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PS_ERR=!errorlevel!"

:: Recharger le PATH (uv s'installe dans plusieurs emplacements possibles)
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv\bin;%PATH%"

if "!PS_ERR!" neq "0" (
    echo.
    echo [ERREUR] L'installation de uv a echoue ^(code !PS_ERR!^).
    echo Installez uv manuellement : https://docs.astral.sh/uv/getting-started/installation/
    goto :error_pause
)

where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] uv est installe mais n'est pas encore accessible dans cette session.
    echo Fermez cette fenetre et relancez launch.bat.
    goto :error_pause
)

echo [OK] uv installe avec succes.
echo.

:uv_ok
for /f "tokens=*" %%v in ('uv --version 2^>nul') do set "UV_VER=%%v"
echo [OK] !UV_VER! detecte.

:: ============================================================
:: 2. Verifier / installer Python 3.13 via uv
::    NOTE : on fixe 3.13 pour eviter les caracteres '<' et '>'
::    qui sont interpretes comme redirections par cmd.exe.
:: ============================================================
echo.
echo [INFO] Verification de Python 3.13...

uv python find 3.13 >nul 2>&1
if not errorlevel 1 goto :python_ok

echo [INFO] Python 3.13 absent - installation en cours ^(peut prendre quelques minutes^)...
uv python install 3.13
if errorlevel 1 (
    echo.
    echo [ERREUR] Impossible d'installer Python 3.13 via uv.
    goto :error_pause
)

:python_ok
echo [OK] Python 3.13 disponible.

:: ============================================================
:: 3. Verifier R / installer les packages requis (MASS, randcorr)
::    rpy2 est une dependance du projet et necessite R sur Windows.
:: ============================================================
where Rscript >nul 2>&1
if not errorlevel 1 goto :r_found

echo.
echo [AVERTISSEMENT] R n'est pas installe ou absent du PATH.
echo   La dependance rpy2 necessite R pour fonctionner.
echo   Telechargez R : https://cran.r-project.org/bin/windows/base/
echo.
echo   Appuyez sur O pour continuer quand meme,
echo   ou sur N pour annuler et installer R d'abord.
echo.

:ask_r
set /p "USER_CHOICE=Continuer sans R ? [O/N] : "
if /i "!USER_CHOICE!"=="O" goto :r_ok
if /i "!USER_CHOICE!"=="N" goto :cancelled
echo Repondez par O ou N.
goto :ask_r

:r_found
:: R est present : installer automatiquement les packages requis (MASS, randcorr)
echo.
echo [INFO] Installation / verification des packages R requis ^(MASS, randcorr^)...
Rscript -e "pkgs <- c('MASS','randcorr'); for(p in pkgs){ if(!requireNamespace(p, quietly=TRUE)){ cat('[INFO] Installation du package R :', p, '\n'); install.packages(p, repos='https://cran.r-project.org', quiet=FALSE) } else { cat('[OK] Package R :', p, 'deja installe\n') }}"
if errorlevel 1 (
    echo.
    echo [AVERTISSEMENT] Echec lors de la verification des packages R.
    echo   Les generateurs bases sur R ^(DDD_BETA^) pourraient ne pas fonctionner.
)

:r_ok

:: ============================================================
:: 4. Synchroniser les dependances (uv sync)
::    --python 3.13 : syntaxe sans '<' / '>' problematiques
:: ============================================================
echo.
echo [INFO] Synchronisation des dependances...
echo   ^(premiere execution = plusieurs minutes^)
echo.

uv sync --python 3.13
if errorlevel 1 (
    echo.
    echo [ERREUR] uv sync a echoue.
    echo Verifiez votre connexion internet et les messages ci-dessus.
    goto :error_pause
)
echo [OK] Dependances synchronisees.

:: ============================================================
:: 5. Lancer l'interface Streamlit
:: ============================================================
echo.
echo [INFO] Demarrage de vote_simulation UI...
echo   Le navigateur va s'ouvrir automatiquement.
echo   Tapez 'stop' ou 's' + Entree pour arreter le serveur.
echo   Vous pouvez aussi fermer l'onglet du navigateur (arret automatique).
echo.

uv run --python 3.13 vote-sim-ui
set "RUN_ERR=!errorlevel!"

if "!RUN_ERR!" neq "0" (
    echo.
    echo [ERREUR] L'interface s'est terminee avec le code !RUN_ERR!.
    goto :error_pause
)

echo.
echo [INFO] Interface fermee proprement.
goto :end

:cancelled
echo.
echo Lancement annule par l'utilisateur.
goto :end

:error_pause
echo.
echo ============================================================
echo  Une erreur s'est produite. Lisez les messages ci-dessus.
echo ============================================================
pause
exit /b 1

:end
endlocal
pause
