function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

temp = X*theta; % dimension: (m*(n+1))*((n+1)*1) = m*1
h = sigmoid(temp); % dimension: m*1
temp_1 = -y.*log(h); % dimension: m*1
temp_0 = (y-ones(size(y))).*log(ones(size(h))-h); % dimension: m*1
temp = temp_0 + temp_1; % dimension: m*1

temp_theta = theta; % dimension: (n+1)*1
temp_theta(1) = 0;
sqr = temp_theta'*temp_theta; % dimension: 1*1; a scalar

J = sum(temp)/m + lambda*sqr/(2*m); % dimension: 1*1; a scalar

temp = h - y; % dimension: m*1
temp = X'*temp; % dimension: ((n+1)*m)*(m*1) = (n+1)*1
grad = temp/m + (lambda/m)*temp_theta; % dimension: (n+1)*1





% =============================================================

end
