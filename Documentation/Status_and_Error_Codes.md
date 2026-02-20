# Error and Status Codes

All of the solvers have an internal status code that is updated as integration is performed. With the exception of the
CySolver class, this code is not generally accessible to the user unless in debug mode. For the CySolver class, you can
access the code via "CySolverInstance.status". 

However, the status will determine what message is produced for each solver.
The currently implemented codes, and their respective messages, are listed below along with some troubleshooting suggestions.

## Status Codes and Messages

- 2: _No Message_
  - This is a status code use for the CySolver indicating that some process, other than integration, is currently being performed. E.g., Interpolation.
- 1: "Integration completed without issue."
  - No obvious issues were encountered. There may still be a problem with the solution but it did not cause a problem during integration.
- 0: "Integration is/was ongoing (perhaps it was interrupted?)."
  - This indicates that integration was on-going but was interrupted before completion.
It is unlikely to see this code unless in debug mode or if the integration was interrupted externally.
- -1: "Error in step size calculation:\n\tRequired step size is less than spacing between numbers."
  - Step sizes are calculated based on local error in the differential equation. This code indicates that the required 
step size to ensure a small solution error is smaller than machine precision. This likely results from a bad set of
initial conditions / optional arguments. And/or that the problem is stiff and not well suited to the selected integration method.
- -2: "Maximum number of steps (set by user) exceeded during integration."
  - The number of steps required during integration exceeded the user set `max_steps` argument.
- -3: "Maximum number of steps (set by system architecture) exceeded during integration."
  - The number of steps required during integration exceeded the maximum number allowed by system architecture (set by 95% of sys.maxsize).
- -4: "Integration has not started."
  - This code is only applicable to the CySolver class. It indicates that a solver instance has been created but the solve() method has not been called.
- -5: "CySolver has been reset."
  - This code is only applicable to the CySolver class. It indicates that a solver instance's reset_state() method has been called but the solve() method has not.
- -6: "Integration never started: y-size is zero."
  - This code indicates that y0 was an empty, size 0, array. There is nothing to integrate.
- -7: "Error in step size calculation:\n\tError in step size acceptance."
  - This is not likely to arise. It comes from a step size that had a problem during calculation but was still accepted.
- -8: "Attribute error."
  - This error indicates that there is a problem with one or more user provided attributes.
