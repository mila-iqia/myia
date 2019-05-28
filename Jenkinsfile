node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: '.testing/jenkins-cache-before.sh'
    sh script: '.testing/install.sh --gpu'
    sh script: '.testing/jenkins-cache-after.sh /home/jenkins/miniconda'
  }
  stage ('Test') {
    sh script: '$HOME/miniconda/bin/pytest --cov-report=term-missing --cov-report=xml  --cov=./ --gpu --junit-xml test-report.xml'
    junit 'test-report.xml'
  }
  stage ('Coverage') {
    sh script: 'ls'
  }
}
