// ============================================================
// chess-sim CI Pipeline
//
// Triggered by: GitHub webhook on every PR push to main.
//
// Stages:
//   1. Build CI Image — rebuild + push when Dockerfile.ci or
//      requirements-ci.txt changed (kaniko, no Docker daemon)
//   2. Detect Changes — diff changed .py files vs base branch
//   3. Lint  — ruff check on changed files only
//   4. Test  — unittest on test files mapped from changed files
//
// Agent: Kubernetes pod defined in jenkins/pod-template.yaml
//   - image: ghcr.io/sombersomni/chess-sim:ci-latest
//   - CUDA disabled; TORCH_DEVICE=cpu
//
// CI image strategy: ci-latest is a mutable tag always pointing
// to the most recent build. When Dockerfile.ci or deps change,
// kaniko rebuilds and pushes ci-latest + ci-sha-<7>. The current
// pipeline run still uses the old image; the NEXT triggered build
// picks up the new one automatically.
// ============================================================

def REGISTRY = 'ghcr.io/sombersomni/chess-sim'

pipeline {
    agent {
        kubernetes {
            yamlFile 'jenkins/pod-template.yaml'
            defaultContainer 'chess-sim'
        }
    }

    options {
        timeout(time: 20, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {
        stage('Build CI Image') {
            steps {
                script {
                    def base = env.CHANGE_TARGET
                        ? "origin/${env.CHANGE_TARGET}"
                        : 'HEAD~1'

                    def ciChanged = sh(
                        script: """
                            git diff --name-only ${base}...HEAD \
                                -- Dockerfile.ci requirements-ci.txt
                        """,
                        returnStdout: true
                    ).trim()

                    if (!ciChanged) {
                        echo 'Dockerfile.ci and requirements-ci.txt unchanged — skipping image build.'
                        env.CI_IMAGE_BUILT = 'false'
                        return
                    }

                    echo "CI image deps changed:\n${ciChanged}"
                    def sha = sh(
                        script: 'git rev-parse --short=7 HEAD',
                        returnStdout: true
                    ).trim()
                    def shaTag  = "ci-sha-${sha}"

                    echo "Building ${REGISTRY}:${shaTag} + ${REGISTRY}:ci-latest"

                    container('kaniko') {
                        sh """
                            /kaniko/executor \
                                --context=\${WORKSPACE} \
                                --dockerfile=Dockerfile.ci \
                                --destination=${REGISTRY}:ci-latest \
                                --destination=${REGISTRY}:${shaTag} \
                                --cache=true \
                                --cache-repo=${REGISTRY}/cache
                        """
                    }
                    env.CI_IMAGE_BUILT = 'true'
                    echo "Pushed ${REGISTRY}:ci-latest and ${REGISTRY}:${shaTag}"
                }
            }
        }

        stage('Detect Changes') {
            steps {
                script {
                    def base = env.CHANGE_TARGET
                        ? "origin/${env.CHANGE_TARGET}"
                        : 'HEAD~1'

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

                    def lintFiles = changed.findAll { f ->
                        fileExists(f)
                    }
                    env.CHANGED_PY = lintFiles.join(' ')

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
