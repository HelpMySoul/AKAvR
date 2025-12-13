# Общий репозиторий AKAvR

💾 **Чтобы сохранить общий репозиторий, используйте следующие команды в git bash**:

```bash
git pull
git add --all
git commit -m "commit"
git push -u origin master
```

Не забудьте настроить appsettings.json перед использованием

**Для установки библиотек**

```bash
{
  "libraries": [
    "tensorflow>=2.10.0",
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "scikit-learn>=1.2.0",
    "joblib>=1.2.0"
  ],
  "libraryVersions": {
    "tensorflow": ">=2.10.0",
    "pandas": ">=1.5.0",
    "numpy": ">=1.23.0",
    "scikit-learn": ">=1.2.0",
    "joblib": ">=1.2.0"
  },
  "extraPipOptions": "--no-cache-dir --timeout=100"
}
```
