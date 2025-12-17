-- Создание таблицы "User_Files"
CREATE TABLE IF NOT EXISTS "User_Files" (
    "Id" SERIAL PRIMARY KEY,
    "FileName" VARCHAR(500) NOT NULL,
    "OriginalFileName" VARCHAR(500),
    "StoredFileName" VARCHAR(1000),
    "FilePath" VARCHAR(1000) NOT NULL,
    "FileSize" BIGINT NOT NULL DEFAULT 0,
    "ContentType" VARCHAR(100),
    "Description" VARCHAR(1000),
    "UploadDate" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "LastAccessed" TIMESTAMP WITH TIME ZONE,
    "UserId" INTEGER NOT NULL
);

-- Внешний ключ к таблице Users
ALTER TABLE "User_Files" 
ADD CONSTRAINT "FK_user_files_Users_UserId" 
FOREIGN KEY ("UserId") 
REFERENCES "Users"("Id") 
ON DELETE CASCADE;

-- Проверка размера файла
ALTER TABLE "User_Files" 
ADD CONSTRAINT "CK_user_files_FileSize" 
CHECK ("FileSize" >= 0);

-- Индексы
CREATE INDEX IF NOT EXISTS "IX_user_files_UserId" ON "User_Files"("UserId");
CREATE INDEX IF NOT EXISTS "IX_user_files_UploadDate" ON "User_Files"("UploadDate");
CREATE INDEX IF NOT EXISTS "IX_user_files_UserId_FileName" ON "User_Files"("UserId", "FileName");

-- Проверка создания
SELECT 'Table "User_Files" created successfully!' as message;