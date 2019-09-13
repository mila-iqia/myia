node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: './ci-install.sh --gpu'
  }
  try {
    stage ('Test') {
      sh script: '$HOME/miniconda/bin/pytest --cov=./ --gpu --junit-xml test-report.xml'
    }
  } finally {
    junit 'test-report.xml'
  }
  stage ('Coverage') {
    sh script: '$HOME/miniconda/bin/pip install codecov'
    sh script: '$HOME/miniconda/bin/coverage combine -a .coverage.*'
    sh script: '$HOME/miniconda/bin/coverage report -m'
    sh script: '$HOME/miniconda/bin/coverage xml'
    withCredentials([string(credentialsId: 'myia_codecov', variable: 'CODECOV_TOKEN')]) {
      sh script: '$HOME/miniconda/bin/codecov'
    }
  }
}
