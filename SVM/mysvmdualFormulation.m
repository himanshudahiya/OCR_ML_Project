function [ alpha ] = mysvmdualFormulation( X, Y, K, C )
    N = size(X, 1);
    H = zeros(N,N);
    for i = 1:N        
        for j=1:N
            H(i,j) = Y(i,1)*Y(j,1)*K(i,j);
        end
    end
    
    f = -ones(N, 1);
    A = [];
    b = [];
    Aeq = Y';
    beq = 0;
    lb = zeros(N, 1);
    ub = C * ones(N, 1);
    'quadprog running'
    alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub);
    'quadprog ended'
    size(alpha)
end