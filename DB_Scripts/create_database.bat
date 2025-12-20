@echo off
REM create_database.bat
REM Batch file to create AKAVER_DB database

echo ========================================
echo Creating AKAVER_DB database...
echo ========================================

SET SQL_FILE=create_database.sql

REM Check if SQL file exists
if not exist "%SQL_FILE%" (
    echo ERROR: File %SQL_FILE% not found!
    pause
    exit /b 1
)

echo Creating database in PostgreSQL...
%PSQL_PATH% -h %PGHOST% -p %PGPORT% -U %PGUSER% -d postgres -f "%SQL_FILE%"

if %errorlevel% equ 0 (
    echo Database AKAVER_DB created successfully!
) else (
    echo Database creation failed!
    exit /b 1
)