function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

temp = X*theta; % dimension: (m*(n+1))*((n+1)*1) = m*1
h = sigmoid(temp); % dimension: m*1
temp_1 = -y.*log(h); % dimension: m*1
temp_0 = (y-ones(size(y))).*log(ones(size(h))-h); % dimension: m*1
temp = temp_0 + temp_1; % dimension: m*1
J = sum(temp)/m; % dimension: 1*1; a scalar

temp = h - y; % dimension: m*1
temp = X'*temp; % dimension: ((n+1)*m)*(m*1) = (n+1)*1
grad = temp/m; % dimension: (n+1)*1


% =============================================================

end