function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
h = X*theta; % dimension: (m*(n+1))*((n+1)*1) = m*1
temp = h - y; % dimension: (m*1)-(m*1) = m*1
J = temp'*temp/(2*m); % dimension: (1*m)*(m*1) = 1*1; a scalar




% =========================================================================

end
