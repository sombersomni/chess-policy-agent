// ============================================================
// chess-sim CI Pipeline
//
// Triggered by: GitHub webhook on every PR push to main.
//
// Stages:
//   1. Lint  — ruff check (exits non-zero on any violation)
//   2. Test  — unittest discover across tests/ (CPU-only)
//
// Agent: Kubernetes pod defined in jenkins/pod-template.yaml
//   - image: YOUR_DOCKERHUB_USER/chess-sim:ci
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
        stage('Lint') {
            steps {
                sh 'python -m ruff check .'
            }
        }

        stage('Test') {
            steps {
                sh 'python -m unittest discover -s tests -p "test_*.py" 2>&1 | tee test-results.txt'
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
