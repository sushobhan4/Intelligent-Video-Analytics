#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <iostream>
#include <string>
#include <vector>

std::wstring GetExecutableDir() {
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(NULL, buffer, MAX_PATH);
    std::wstring path(buffer);
    size_t pos = path.find_last_of(L"\\/");
    return (pos != std::wstring::npos) ? path.substr(0, pos) : L"";
}

bool FileExists(const std::wstring& path) {
    DWORD dwAttrib = GetFileAttributesW(path.c_str());
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

int wmain(int argc, wchar_t* argv[]) {
    // Ensure console output is in UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // Set performance environment variables for CUDA and Python
    SetEnvironmentVariableW(L"PYTORCH_CUDA_ALLOC_CONF", L"expandable_segments:True");
    SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
    SetEnvironmentVariableW(L"PYTHONUNBUFFERED", L"1");

    std::wstring exeDir = GetExecutableDir();
    std::wstring scriptPath = exeDir + L"\\test_qwen3_pipeline.py";

    // Locate Python 3.13 executable
    std::wstring pythonPath = L"C:\\Users\\halda\\AppData\\Local\\Programs\\Python\\Python313\\python.exe";
    if (!FileExists(pythonPath)) {
        pythonPath = L"python.exe"; // Fallback to PATH
    }

    if (!FileExists(scriptPath)) {
        // Fallback: check current directory
        scriptPath = L"test_qwen3_pipeline.py";
    }

    // Build the command line string
    std::wstring cmdLine = L"\"" + pythonPath + L"\" \"" + scriptPath + L"\"";

    for (int i = 1; i < argc; ++i) {
        cmdLine += L" ";
        std::wstring arg = argv[i];
        if (arg.find(L' ') != std::wstring::npos) {
            cmdLine += L"\"" + arg + L"\"";
        } else {
            cmdLine += arg;
        }
    }

    // Launch the compiled pipeline process
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    std::vector<wchar_t> cmdLineBuf(cmdLine.begin(), cmdLine.end());
    cmdLineBuf.push_back(L'\0');

    BOOL success = CreateProcessW(
        NULL,
        cmdLineBuf.data(),
        NULL,
        NULL,
        TRUE,
        0,
        NULL,
        exeDir.c_str(),
        &si,
        &pi
    );

    if (!success) {
        std::wcerr << L"[!] Failed to launch CCTV Analyzer. Error code: " << GetLastError() << std::endl;
        return 1;
    }

    // Wait for the analysis process to complete
    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return static_cast<int>(exitCode);
}
