function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X*theta; % dimension: (m*(n+1))*((n+1)*1) = m*1
temp = h - y; % dimension: (m*1)-(m*1) = m*1

temp_theta = theta; % dimension: (n+1)*1
temp_theta(1) = 0;
sqr = temp_theta'*temp_theta; % dimension: 1*1; a scalar

J = temp'*temp/(2*m) + lambda*sqr/(2*m); % dimension: 1*1; a scalar

temp = X'*temp; % dimension: ((n+1)*m)*(m*1) = (n+1)*1
grad = temp/m + (lambda/m)*temp_theta; % dimension: (n+1)*1







% =========================================================================

grad = grad(:);

end
