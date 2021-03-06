function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
A_1 = [ones(m, 1) X]; % dimension: 1*(n+1)
Z_2 = A_1*Theta1'; % dimension: (1*(n+1))*((n+1)*25) = 1*25
A_2 = sigmoid(Z_2); % dimension: 1*25
A_2 = [ones(m, 1) A_2]; % dimension: 1*26
Z_3 = A_2*Theta2'; % dimension: (1*26)*(26*10) = 1*10
A_3 = sigmoid(Z_3); % dimension: 1*10
[x, ix] = max(A_3, [], 2); % take rowwise max and max indices
p = ix; % save max indices as predictions







% =========================================================================


end
