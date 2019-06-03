node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: '.testing/install.sh --gpu'
  }
  stage ('Test') {
    sh script: '$HOME/miniconda/bin/pytest --cov-report=term-missing --cov-report=xml  --cov=./ --gpu --junit-xml test-report.xml'
    junit 'test-report.xml'
  }
  stage ('Coverage') {
    configFileProvider([configFile('2168de59-9641-4ad3-95bb-28e0a131815e')]) {
      sh script: 'ls'
      sh script: 'curl -s https://codecov.io/bash | bash -- -t `cat coverage.token`'
    }
  }
}
