Last name of Student 1: Lu
First name of Student 1: Wesley
Email of Student 1: wesley_lu@ucsb.edu
GradeScope account name of Student 1: wesley_lu@umail.ucsb.edu
Last name of Student 2: Yang
First name of Student 2: Shu
Email of Student 2: syang01@ucsb.edu
GradeScope account name of Student 2: syang01@umail.ucsb.edu


----------------------------------------------------------------------------
Report for Question 1 


Parallel time for n=4K, t=1K,  4x128  threads (test 9c)
15.037681

Parallel time for n=4K, t=1K,  8x128  threads (test 9b)
7.532955 seconds

Parallel time for n=4K, t=1K,  16x128 threads (test 9a)
3.767483 seconds

Parallel time for n=4K, t=1K,  32x128 threads (test 9)
4.504365 seconds

Why there is less speedup improvement  towards 32x128 threads?
Amdahl's law: Possible speedup is limited by available parallelism

----------------------------------------------------------------------------
Report for Question 2 

Parallel time for n=4K, t=1K,  8x128  threads with shared memory (test 10a)
7.182028 seconds

Parallel time for n=4K, t=1K,  32x128 threads with shared memory (test 10)
4.276883

Explain the reason if you see an improvement compared to Question 1 code performance.  
There is a slight improvement compared to the code in Question 1 because shared memory bandwith is faster than global memory bandwith

Explain the reason if you see less improvement when the total  number of threads increases. 
We did not see less improvement when the total number of threads increases.

----------------------------------------------------------------------------
Report for Question 3 
The default number of asynchronous iterations is 5 in a batch.
Parallel time and the number of actual iterations executed  for n=4K, t=1K, 8x128  threads with asynchronous Gauss Seidel (test 11a)
.187885 seconds

Parallel time and the number of actual ierations executed  for n=4K, t=1K,  32x128 threads with asynchronous Gauss Seidel (test 11)
.467079 seconds

Does the above Gauss Seidel -Seidel method converge faster than the Jacobi method in this case?  
Compare performance with code of Question 1 in the above setting and explain the difference.

Yes, Gauss-Seidel method converges faster than Jacobi because it uses updated values as soon as they are available. The performance of this code compared to the code referenced in Question 1 is much faster.

