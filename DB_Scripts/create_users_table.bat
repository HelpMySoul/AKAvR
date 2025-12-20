@echo off
REM create_users_table.bat
REM Batch file to create Users table

echo ========================================
echo Creating Users table in AKAVER_DB...
echo ========================================

SET SQL_FILE=create_users_table.sql

REM Check if SQL file exists
if not exist "%SQL_FILE%" (
    echo ERROR: File %SQL_FILE% not found!
    pause
    exit /b 1
)

echo Waiting for database to be ready...
timeout /t 2 /nobreak >nul

echo Creating table and test data in AKAVER_DB...
%PSQL_PATH% -h %PGHOST% -p %PGPORT% -d %PGDATABASE% -U %PGUSER% -f "%SQL_FILE%"

if %errorlevel% equ 0 (
    echo ✓ Users table created successfully!
) else (
    echo ✗ Table creation failed!
    echo This usually means:
    echo 1. Database AKAVER_DB was not created in previous step
    echo 2. Connection issues to PostgreSQL
    echo 3. Database name is incorrect
    exit /b 1
)