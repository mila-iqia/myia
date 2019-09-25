node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: './ci-install.sh --gpu'
  }
  try {
    stage ('Test') {
      sh script: '$HOME/miniconda/bin/pytest --cov=./ --cov-report= --gpu --junit-xml test-report.xml'
    }
  } finally {
    junit 'test-report.xml'
  }
  stage ('Coverage') {
    withEnv(['PATH+CONDA=/home/jenkins/miniconda/bin']) {
      sh script: './cov.sh'
      sh script: 'coverage xml'
      sh script: 'pip install codecov'
      withCredentials([string(credentialsId: 'myia_codecov', variable: 'CODECOV_TOKEN')]) {
        sh script: 'codecov'
      }
    }
  }
}
