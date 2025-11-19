------------------------------------------
-- create_users_table.sql
-- Script to create Users table
------------------------------------------

-- First make sure we're connected to the right database
SELECT current_database();

-- Create Users table
CREATE TABLE IF NOT EXISTS "Users" (
    "Id" SERIAL PRIMARY KEY,
    "Email" VARCHAR(255) NOT NULL UNIQUE,
    "Username" VARCHAR(100) NOT NULL UNIQUE,
    "Password" VARCHAR(255) NOT NULL,
    "CreatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "UpdatedAt" TIMESTAMP WITH TIME ZONE NULL,
    "IsActive" BOOLEAN DEFAULT TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS "IX_Users_Email" ON "Users"("Email");
CREATE INDEX IF NOT EXISTS "IX_Users_Username" ON "Users"("Username");

-- Show results
SELECT 'Users table created successfully!' as message;
SELECT count(*) as "Total users" FROM "Users";
SELECT "Id", "Email", "Username", "IsActive" FROM "Users";