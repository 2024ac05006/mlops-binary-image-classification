pipeline {
    agent any

    environment {
        // Change to your actual GitHub username (must be lowercase)
        GITHUB_USER = '2024ac05006'
        IMAGE_NAME = "ghcr.io/${env.GITHUB_USER}/cats-dogs-service"
        IMAGE_TAG = "v${env.BUILD_ID}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv ci_venv
                . ci_venv/bin/activate
                pip install --upgrade pip
                pip install pytest
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
                pip install -r requirements-serve.txt
                '''
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh '''
                . ci_venv/bin/activate
                pytest tests/ -v
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest ."
            }
        }

        stage('Publish to GHCR') {
            steps {
                // Securely inject Jenkins credentials for docker login
                withCredentials([usernamePassword(credentialsId: 'github-cr-creds', usernameVariable: 'GH_USER', passwordVariable: 'GH_PAT')]) {
                    sh 'echo $GH_PAT | docker login ghcr.io -u $GH_USER --password-stdin'
                    sh "docker push ${IMAGE_NAME}:${IMAGE_TAG}"
                    sh "docker push ${IMAGE_NAME}:latest"
                }
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                sh 'kubectl apply -f k8s/deployment.yaml'
                sh 'kubectl apply -f k8s/service.yaml'
                sh 'kubectl rollout restart deployment cats-dogs-deployment'
                // Increased timeout to 300s to accommodate image download time
                sh 'kubectl rollout status deployment/cats-dogs-deployment --timeout=300s'
            }
        }
        stage('Smoke Tests') {
            steps {
                sh '''
                . ci_venv/bin/activate
                pip install requests Pillow
                python scripts/smoke_test.py
                '''
            }
        }        
    }

    post {
        always {
            sh 'rm -rf ci_venv'
        }
        success {
            echo "CI Pipeline Succeeded! Image published to GHCR: ${IMAGE_NAME}:${IMAGE_TAG}"
        }
    }
}