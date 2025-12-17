@echo off
REM setup_project.bat
REM Main batch file to run database and table creation scripts

setlocal

echo ========================================
echo AKAvR Project Database Setup
echo PostgreSQL 18 
echo Password: postgres
echo ========================================
echo.

REM Load configuration
if not exist "config.bat" (
    echo ERROR: config.bat not found!
    echo Please make sure config.bat is in the same folder
    pause
    exit /b 1
)
call config.bat

echo Step 1: Creating database...
call create_database.bat

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Database creation failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Creating users table...
call create_users_table.bat

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Table creation failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Creating user files table...
call create_user_files_table.bat

if %errorlevel% neq 0 (
    echo.
    echo ERROR: User files table creation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo SETUP COMPLETED SUCCESSFULLY!
echo ========================================
echo Database: %PGDATABASE%
echo Table: Users
echo Your application is ready to use!
echo ========================================
echo.

pause
endlocal