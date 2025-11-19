@echo off
REM run_sql_script.bat
REM Батовый файл для запуска SQL скрипта в PostgreSQL

setlocal

REM Настройки подключения к базе данных
SET PGHOST=localhost
SET PGPORT=5432
SET PGDATABASE=AKAvR_DB
SET PGUSER=postgres
SET PGPASSWORD=postgres

REM Путь к SQL файлу
SET SQL_FILE=create_users_table.sql

REM Проверка существования SQL файла
if not exist "%SQL_FILE%" (
    echo Ошибка: Файл %SQL_FILE% не найден!
    pause
    exit /b 1
)

echo ========================================
echo Запуск SQL скрипта: %SQL_FILE%
echo База данных: %PGDATABASE%
echo Хост: %PGHOST%:%PGPORT%
echo ========================================

REM Запуск SQL скрипта с помощью psql
psql -h %PGHOST% -p %PGPORT% -d %PGDATABASE% -U %PGUSER% -f "%SQL_FILE%"

REM Проверка результата выполнения
if %errorlevel% equ 0 (
    echo ========================================
    echo SQL скрипт успешно выполнен!
    echo ========================================
) else (
    echo ========================================
    echo Ошибка при выполнении SQL скрипта!
    echo ========================================
)

pause
endlocal