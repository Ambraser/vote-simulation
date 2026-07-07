@echo off
setlocal EnableDelayedExpansion

title vote_simulation UI — Lancement

echo ============================================================
echo  vote_simulation UI — Lanceur Windows
echo ============================================================
echo.

:: ── Localiser le dossier du script (chemin du .bat lui-meme) ────────────────
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: ============================================================
:: 1. Verifier / installer uv
:: ============================================================
where uv >nul 2>&1
if errorlevel 1 (
    echo [INFO] uv non trouve. Tentative d'installation via PowerShell...
    echo.
    powershell -ExecutionPolicy Bypass -Command ^
        "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo.
        echo [ERREUR] Impossible d'installer uv automatiquement.
        echo Installez uv manuellement depuis : https://docs.astral.sh/uv/getting-started/installation/
        echo Puis relancez ce fichier.
        goto :error_pause
    )
    :: Recharger le PATH pour que uv soit visible
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ERREUR] uv vient d'etre installe mais n'est pas accessible dans ce terminal.
        echo Fermez cette fenetre, puis relancez launch.bat.
        goto :error_pause
    )
    echo [OK] uv installe avec succes.
    echo.
)

:: ── Afficher la version de uv ───────────────────────────────────────────────
for /f "tokens=*" %%v in ('uv --version 2^>nul') do set "UV_VER=%%v"
echo [OK] %UV_VER% detecte.

:: ============================================================
:: 2. Verifier que Python 3.12+ est disponible via uv
:: ============================================================
echo.
echo [INFO] Verification de Python (3.12-3.14)...
uv python find ">=3.12,<3.15" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Python 3.12-3.14 absent — installation en cours (peut prendre quelques minutes)...
    uv python install 3.13
    if errorlevel 1 (
        echo.
        echo [ERREUR] Impossible d'installer Python via uv.
        goto :error_pause
    )
)
echo [OK] Python trouve.

:: ============================================================
:: 3. Avertissement rpy2 (necessite R sur Windows)
:: ============================================================
where Rscript >nul 2>&1
if errorlevel 1 (
    echo.
    echo [AVERTISSEMENT] R n'est pas installe ou absent du PATH.
    echo   La dependance rpy2 requiert R sur Windows.
    echo   Telechargez R depuis : https://cran.r-project.org/bin/windows/base/
    echo   Si vous n'utilisez pas les fonctionnalites R, l'UI peut fonctionner
    echo   partiellement mais des erreurs d'importation sont possibles.
    echo.
    choice /c OA /m "Continuer quand meme ? [O]ui / [A]nnuler"
    if errorlevel 2 goto :cancelled
)

:: ============================================================
:: 4. Installer / synchroniser les dependances (uv sync)
:: ============================================================
echo.
echo [INFO] Synchronisation des dependances (premiere execution = quelques minutes)...
uv sync --python ">=3.12,<3.15"
if errorlevel 1 (
    echo.
    echo [ERREUR] uv sync a echoue. Verifiez votre connexion internet et les logs ci-dessus.
    goto :error_pause
)
echo [OK] Dependances installees.

:: ============================================================
:: 5. Lancer l'interface Streamlit
:: ============================================================
echo.
echo [INFO] Demarrage de vote_simulation UI dans le navigateur...
echo   Appuyez sur Ctrl+C dans cette fenetre pour arreter le serveur.
echo.

uv run --python ">=3.12,<3.15" vote-sim-ui
if errorlevel 1 (
    echo.
    echo [ERREUR] L'interface Streamlit s'est terminee avec une erreur.
    goto :error_pause
)

echo.
echo [INFO] Interface fermee proprement.
goto :end

:cancelled
echo.
echo Lancement annule.
goto :end

:error_pause
echo.
echo ============================================================
echo  Une erreur s'est produite. Lisez les messages ci-dessus.
echo  Appuyez sur une touche pour fermer cette fenetre.
echo ============================================================
pause >nul
exit /b 1

:end
endlocal
