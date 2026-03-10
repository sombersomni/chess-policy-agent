// Jenkins Groovy script to create chess-sim-pipeline job
// Run via: Jenkins > Manage Jenkins > Script Console

import jenkins.model.*
import org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition
import hudson.plugins.git.BranchSpec
import hudson.plugins.git.UserRemoteConfig
import hudson.plugins.git.GitSCM

def jobName = "chess-sim-pipeline"
def gitUrl = "https://github.com/sombersomni/chess-sim.git"
def branch = "*/main"
def scriptPath = "Jenkinsfile"

def jenkins = Jenkins.get()

// Check if job already exists
if (jenkins.getItem(jobName) != null) {
    println "Job '$jobName' already exists. Deleting..."
    jenkins.getItem(jobName).delete()
}

// Create new pipeline job
def job = jenkins.createProject(org.jenkinsci.plugins.workflow.job.WorkflowJob, jobName)
job.setDescription("Chess-sim CI Pipeline - Lint and Test stages")

// Configure SCM
def scm = new GitSCM([
    new UserRemoteConfig(gitUrl, null, null, null)
])
scm.branches = [new BranchSpec(branch)]
job.setDefinition(new CpsScmFlowDefinition(scm, scriptPath))

job.save()
println "Job '$jobName' created successfully!"
