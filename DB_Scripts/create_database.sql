------------------------------------------
-- create_database.sql
-- Script to create AKAVER_DB database
------------------------------------------

-- Create database if it doesn't exist
CREATE DATABASE "AKAVER_DB";

-- Display success message
\echo 'Database AKAVER_DB created successfully!'

-- Show all databases to verify
SELECT datname as "Database Name" 
FROM pg_database 
WHERE datname IN ('postgres', 'AKAVER_DB')
ORDER BY datname;