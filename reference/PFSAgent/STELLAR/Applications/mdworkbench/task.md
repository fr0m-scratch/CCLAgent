# MD-Workbench Optimize Storage Performance

Tune the -P parameter to ensure a working set of at least 1 million files, verifying that data is indeed being written to storage. To achieve this, follow a three-step process:

* Precreate the environment using -1.
* Clear the cache by either cleaning the OS cache or running an IOR write operation to flush out the cache.
* Run phases -2 and -3 to generate the working set and measure performance.

When configuring the working set, keep in mind that the total number of objects is calculated as -D * -P. You can use the -R option for verification purposes and to demonstrate variance in performance.

Goal: Minimize the op-max value from the output to achieve optimal storage performance.

## Simple execution example
> md-workbench -1
> mpiexec -np 2 ./md-workbench -2 -3