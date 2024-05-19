# Landscapes_Prediction

## Installation

Make sure to have Python v3.9 or higher (v3.9.12 recommended)

After cloning the repository, create a new environment by executing the command inside the cloned directory:
'''
python -m venv env
'''
Now activate the environment and install the required packages by running the command:

'''
env\Scripts\activate
pip install -r requirement.txt
'''
```
>>> employee = pd.read_csv('data/employee')
>>> max_dept_salary = employee.groupby('DEPARTMENT')['BASE_SALARY'].max()
```
Finally, run the command to deploy the flask app over localhost:5000
'''
python app.py
'''
