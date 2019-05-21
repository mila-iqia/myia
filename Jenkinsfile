node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
  cache(caches: [[$class: 'ArbitraryFileCache', excludes: '', includes: '**/*', path: '/home/jenkins/miniconda']], maxCacheSize: 2000) {
      sh script: '.testing/install.sh --gpu'
    }
  }
  stage ('Test') {
    sh script: '$HOME/miniconda/bin/pytest --cov=./ --gpu'
  }
  stage ('Coverage') {
    sh script: 'codecov'
  }
}
