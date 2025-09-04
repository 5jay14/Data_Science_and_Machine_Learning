import pandas as pd

sal = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\04"
                  r"-Pandas-Exercises\Salaries.csv")


# print(sal.columns)  # Identify how many columns
# print(sal.head()) # limits top 5 records
# print(sal.info()) # Use the .info() method to find out how many entries there are and all the other info
# print(sal.describe().transpose())

# What is the average BasePay ?
# print(sal['BasePay'].mean())

# What is the highest amount of OvertimePay in the dataset ? **
# print(sal['OvertimePay'].max())

#  What is the job title of JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match
#  up (there is also a lowercase Joseph Driscoll
# print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle'])


# How much does JOSEPH DRISCOLL make (including benefits)?
# print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL'] ['TotalPayBenefits'])

# What is the name of highest paid person (including benefits)?
# print(sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName'])
# print(sal.loc[sal['TotalPayBenefits'].idxmax()])

# What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or
# she is paid?
# print(sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]['EmployeeName'])
# print(sal.loc[sal['TotalPayBenefits'].idxmin()])

# What was the average (mean) BasePay of all employees per year? (2011-2014) ?
# print(sal.groupby('Year')['BasePay'].mean())


# How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurrence in 2013?)
# print(sum(sal[sal['Year']==2013]['JobTitle'].value_counts()==1))
# print(sal['JobTitle'].nunique())  # this gives the total jobs that were unique
# print(sal['JobTitle'].unique())  #this gives the names of the jobs that were unique

# What are the top 5 most common jobs?
# print(sal['JobTitle'].value_counts().head()) # sorting is already set to descending

# How many people have the word Chief in their job title? (This is pretty tricky)
def chief_string(title):
    if 'chief' in title.lower():
        return True
    else:
        return False


print(sum(sal['JobTitle'].apply(lambda x: chief_string(x))))

# Bonus: Is there a correlation between length of the Job Title string and Salary?
