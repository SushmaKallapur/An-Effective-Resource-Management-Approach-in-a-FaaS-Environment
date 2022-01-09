# An-Effective-Resource-Management-Approach-in-a-FaaS-Environment
Analyzed techniques which delivers optimal solution for resource management in a FaaS environment

------
Basic steps to run project :
------
1. Pre-requisite: <br />
    1. Free Tier Account of AWS.
    2. Docker installed - v20.x+ .
    3. NPM installed - vNode12.x+. <br />
2. AWS config set up:
   1. Click on add-User under IAM-> Users.
   2. Add a username and add access under access Type.
   3. Add the administration policy.
   4. Create the user.
   5. Need to install AWS CLI
      1. Windows: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
      2. Mac: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html#cliv2-mac-install-cmd
   6. check the version:
      1. aws --version.
   7. To configure aws, type in below command:
      1. aws configure.
3. Serverless Set up:
   1. Install serverless CLI
      1. npm install -f serverless.
   2. serverless --version.
   3. Create serverless project:
      1. serverless create --template aws-python-docker --path {ProjectName}.
4. Override created project file with given project folder in that directory.
5. Now, Run below command in the project directory:
   1. To deploy the function on AWS as lambda function:
      1. serverless deploy 
   2. Invoke lambda function
      1. sudo serverless invoke -f MOGA-algo --log
   
----
Results : 
-----
After running above steps, we can see results as below with some graphs.


1. NSGAII :
   1. Pareto front evaluations:
      ![alt text](https://github.com/SushmaKallapur/An-Effective-Resource-Management-Approach-in-a-FaaS-Environment/blob/main/Readme_Files/Pareto_Front_Nsga2.png)
   2. 3d visualization of evaluation objects:
      ![alt_text](https://github.com/SushmaKallapur/An-Effective-Resource-Management-Approach-in-a-FaaS-Environment/blob/main/Readme_Files/3d_evaluation_objects_nsga2.png)
   3. HyperVolume Comparison:
      ![alt_text](https://github.com/SushmaKallapur/An-Effective-Resource-Management-Approach-in-a-FaaS-Environment/blob/main/Readme_Files/NSGAII_HV_values.png)
      
2. NSGA III:
   1. Pareto front Evaluations:
      ![alt_text](https://github.com/SushmaKallapur/An-Effective-Resource-Management-Approach-in-a-FaaS-Environment/blob/main/Readme_Files/pareto_front_nsga3.png)
   2. 3d visualization of evaluation objects:
      ![alt_text](https://github.com/SushmaKallapur/An-Effective-Resource-Management-Approach-in-a-FaaS-Environment/blob/main/Readme_Files/3d_evaluation_objects_nsga3.png)
   3. HyperVolume Comparison: 
      ![alt_text](https://github.com/SushmaKallapur/An-Effective-Resource-Management-Approach-in-a-FaaS-Environment/blob/main/Readme_Files/NSGAIII_HV_Values.png)

-----
Note :
------
1. There are many dependencies which are required to run this project.
2. We executed the project on M1 chip Mac. We are not sure how it will handle on other types of devices.
3. We have added all the dependencies in the requirement.txt file.



