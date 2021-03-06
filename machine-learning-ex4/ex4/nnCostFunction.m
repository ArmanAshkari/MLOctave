function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Add ones to the X data matrix
A_1 = [ones(m, 1) X]; % dimension: m*(n+1)
Z_2 = A_1 * Theta1'; % dimension: (m*(n+1))*((n+1)*25) = m*25
A_2 = sigmoid(Z_2); % dimension: m*25
A_2 = [ones(m, 1) A_2]; % dimension: m*26
Z_3 = A_2 * Theta2'; % dimension: (m*26)*(26*10) = m*10
A_3 = sigmoid(Z_3); % dimension: m*10
% Format y matrix from labels
temp_y = eye(num_labels); % dimension: 10*10
temp_y = temp_y(y, :); % dimension: m*10
% Cost calculation
temp = - temp_y .* log(A_3) - (ones(size(temp_y)) - temp_y) .* log(ones(size(A_3)) - A_3);
J = sum(sum(temp)) / m;
% Add regularization
temp_Theta1 = Theta1;
temp_Theta1(:, 1) = 0; % Do not regularize first parameter
temp_Theta2 = Theta2;
temp_Theta2(:,1) = 0; % Do not regularize first parameter
J = J + (sum(sum(temp_Theta1 .* temp_Theta1)) + sum(sum(temp_Theta2 .* temp_Theta2))) * lambda / (2*m);


delta3 = A_3 - temp_y; % dimension m*10
delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(Z_2); % dimension m*25
% Ignore the first column of Theta2 (bias term does not propagate backwards). 

% It was verified that we can calculate DELTAs for all training examples in this process
DELTA2 = delta3' * A_2; % dimension 10*26
DELTA1 = delta2' * A_1; % dimension 25*(n+1)

D2 = DELTA2 / m + temp_Theta2 * lambda / m; % dimension 10*26
D1 = DELTA1 / m + temp_Theta1 * lambda / m; % dimension 25*(n+1)

Theta1_grad = D1; % dimension 25*(n+1)
Theta2_grad = D2; % dimension 10*26












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
