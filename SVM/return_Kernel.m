function [ kernel ] =return_Kernel(X,gamma )
    [N D] = size(X);
    kernel = zeros(N,N);
    for i = 1:N    
        for j = 1:N
            kernel(i,j) = (gamma *( X(i, :)*transpose( X(j, :)))).^4;
        end    
    end
end


