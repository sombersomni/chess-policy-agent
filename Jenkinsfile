// ============================================================
// chess-sim CI Pipeline
//
// Triggered by: GitHub webhook on every PR push to main.
//
// Stages:
//   1. Detect Changes — diff changed .py files vs base branch
//   2. Lint  — ruff check on changed files only
//   3. Test  — unittest on test files mapped from changed files
//
// Agent: Kubernetes pod defined in jenkins/pod-template.yaml
//   - image: ghcr.io/sombersomni/chess-sim:ci-latest
//   - CUDA disabled; TORCH_DEVICE=cpu
// ============================================================

pipeline {
    agent {
        kubernetes {
            yamlFile 'jenkins/pod-template.yaml'
            defaultContainer 'chess-sim'
        }
    }

    options {
        // Kill the build if it hangs (PyTorch install, etc.)
        timeout(time: 20, unit: 'MINUTES')
        // Prevent duplicate builds piling up on the same branch
        disableConcurrentBuilds()
    }

    stages {
        stage('Detect Changes') {
            steps {
                script {
                    // PR builds: CHANGE_TARGET is the base branch (e.g. main).
                    // Direct pushes: fall back to comparing against HEAD~1.
                    def base = env.CHANGE_TARGET ? "origin/${env.CHANGE_TARGET}" : 'HEAD~1'

                    def raw = sh(
                        script: "git diff --name-only ${base}...HEAD -- '*.py'",
                        returnStdout: true
                    ).trim()

                    if (!raw) {
                        echo 'No Python files changed — skipping lint and test.'
                        env.CHANGED_PY   = ''
                        env.CHANGED_TESTS = ''
                        return
                    }

                    def changed = raw.split('\n') as List

                    // Lint: all changed .py files that still exist on disk.
                    def lintFiles = changed.findAll { f ->
                        fileExists(f)
                    }
                    env.CHANGED_PY = lintFiles.join(' ')

                    // Test: map each changed source file to its test file.
                    //   chess_sim/.../foo.py  →  tests/.../test_foo.py  (first match wins)
                    //   tests/test_foo.py     →  kept as-is
                    def testFiles = [] as Set
                    changed.each { f ->
                        if (f.startsWith('tests/') && f.endsWith('.py') && fileExists(f)) {
                            testFiles << f
                        } else {
                            def stem = f.tokenize('/')[-1].replace('.py', '')
                            def candidate = "tests/test_${stem}.py"
                            if (fileExists(candidate)) {
                                testFiles << candidate
                            }
                        }
                    }
                    env.CHANGED_TESTS = testFiles.join(' ')

                    echo "Changed source files : ${env.CHANGED_PY}"
                    echo "Mapped test files    : ${env.CHANGED_TESTS}"
                }
            }
        }

        stage('Lint') {
            when { expression { env.CHANGED_PY?.trim() } }
            steps {
                sh "python -m ruff check ${env.CHANGED_PY}"
            }
        }

        stage('Test') {
            when { expression { env.CHANGED_TESTS?.trim() } }
            steps {
                sh "python -m unittest ${env.CHANGED_TESTS.replace('/', '.').replace('.py', '').split(' ').join(' ')} 2>&1 | tee test-results.txt"
            }
            post {
                always {
                    archiveArtifacts artifacts: 'test-results.txt', allowEmptyArchive: true
                }
            }
        }
    }

    post {
        success {
            echo 'CI passed: lint clean, all tests green.'
        }
        failure {
            echo 'CI failed. Check Lint output or test-results.txt for details.'
        }
    }
}
