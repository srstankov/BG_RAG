# -*- mode: python ; coding: utf-8 -*-

added_binary_files = [

("venv/Lib/site-packages/llama_cpp/lib", "llama_cpp/lib"),

]

added_datas_files = [

("db_bm25s_chunks_index", "db_bm25s_chunks_index"),
("focus_news", "focus_news"),
("images", "images"),
("db_chunks_and_embeddings.pkl", "."),
("faiss_vector_db.index", "."),
("icon.ico", "."),
("options_config.json", "."),
("venv/Lib/site-packages/en_core_web_sm", "en_core_web_sm"),
("venv/Lib/site-packages/en_core_web_sm-3.7.1.dist-info", "en_core_web_sm-3.7.1.dist-info"),
("venv/Lib/site-packages/llama_cpp", "llama_cpp"),
("venv/Lib/site-packages/llama_cpp_python-0.3.4.dist-info", "llama_cpp_python-0.3.4.dist-info"),

]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=added_binary_files,
    datas=added_datas_files,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BG_RAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BG_RAG_2',
)
