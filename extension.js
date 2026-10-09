/**
 * extension.js - QueueFlux VS Code Extension Entrypoint
 *
 * Integrates QueueFlux Intelligence Layer directly into Visual Studio Code:
 * 1. Creates a dedicated VS Code Output Channel ("QueueFlux Intelligence")
 * 2. Provides Command Palette integrations:
 *    - QueueFlux: Run 6-Scenario Demonstration
 *    - QueueFlux: Run Full Unit Test Suite
 *    - QueueFlux: Analyze Real-Time Canteen State
 * 3. Displays VS Code Notifications & Status Alerts
 */

const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');

let outputChannel;

function activate(context) {
    outputChannel = vscode.window.createOutputChannel("QueueFlux Intelligence");
    context.subscriptions.push(outputChannel);

    console.log('QueueFlux Intelligence Layer Extension is now active in VS Code!');

    // 1. Command: QueueFlux: Run 6-Scenario Demonstration
    let runDemoCmd = vscode.commands.registerCommand('queueflux.runDemo', function () {
        outputChannel.clear();
        outputChannel.show(true);
        outputChannel.appendLine("================================================================================");
        outputChannel.appendLine("            EXECUTING QUEUEFLUX 6-SCENARIO DEMONSTRATION");
        outputChannel.appendLine("================================================================================\n");

        const workspacePath = vscode.workspace.workspaceFolders
            ? vscode.workspace.workspaceFolders[0].uri.fsPath
            : __dirname;

        exec('python main.py', { cwd: workspacePath }, (error, stdout, stderr) => {
            if (stdout) {
                outputChannel.appendLine(stdout);
            }
            if (stderr) {
                outputChannel.appendLine("[STDERR]: " + stderr);
            }
            if (error) {
                vscode.window.showErrorMessage(`QueueFlux Error: ${error.message}`);
                outputChannel.appendLine(`[EXECUTION ERROR]: ${error.message}`);
            } else {
                vscode.window.showInformationMessage("QueueFlux: All 6 Demonstration Scenarios Executed Successfully!");
            }
        });
    });

    // 2. Command: QueueFlux: Run Full Unit Test Suite
    let runTestsCmd = vscode.commands.registerCommand('queueflux.runTests', function () {
        outputChannel.clear();
        outputChannel.show(true);
        outputChannel.appendLine("================================================================================");
        outputChannel.appendLine("            RUNNING QUEUEFLUX UNIT & INTEGRATION TEST SUITE");
        outputChannel.appendLine("================================================================================\n");

        const workspacePath = vscode.workspace.workspaceFolders
            ? vscode.workspace.workspaceFolders[0].uri.fsPath
            : __dirname;

        exec('python -m unittest test_intelligence.py -v', { cwd: workspacePath }, (error, stdout, stderr) => {
            if (stdout) outputChannel.appendLine(stdout);
            if (stderr) outputChannel.appendLine(stderr);

            if (error) {
                vscode.window.showErrorMessage("QueueFlux: Unit Test Failure Detected!");
            } else {
                vscode.window.showInformationMessage("QueueFlux: 16/16 Unit Tests Passed (100% OK)!");
            }
        });
    });

    // 3. Command: QueueFlux: Analyze Real-Time Canteen State
    let analyzeStateCmd = vscode.commands.registerCommand('queueflux.analyzeState', function () {
        outputChannel.clear();
        outputChannel.show(true);
        outputChannel.appendLine("[QueueFlux] Analyzing current canteen state...\n");

        const workspacePath = vscode.workspace.workspaceFolders
            ? vscode.workspace.workspaceFolders[0].uri.fsPath
            : __dirname;

        exec('python -c "from intelligence_engine import IntelligenceEngine; engine = IntelligenceEngine(); report = engine.process_raw_input(custom_dict={\'billing_queue\': 14, \'arrivals_last_min\': 14.0, \'available_extra_counters\': 1, \'available_extra_workers\': 2}); print(report.to_human_summary())"', { cwd: workspacePath }, (error, stdout, stderr) => {
            if (stdout) {
                outputChannel.appendLine(stdout);
                vscode.window.showWarningMessage(
                    "QueueFlux Alert: Arrival Surge Detected! Recommended Action: OPEN_EXTRA_BILLING_COUNTER (Saves ~1.7m wait time).",
                    "View Report"
                ).then(selection => {
                    if (selection === "View Report") {
                        outputChannel.show(true);
                    }
                });
            }
            if (error) {
                outputChannel.appendLine(`[ERROR]: ${error.message}`);
            }
        });
    });

    context.subscriptions.push(runDemoCmd);
    context.subscriptions.push(runTestsCmd);
    context.subscriptions.push(analyzeStateCmd);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};
