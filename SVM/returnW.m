function [w] = returnW(alpha, K, Y)

    w = 0;
    
    N = size(K, 1);
    
    for i = 1:N
        x = sum(alpha .* Y .* K(:, i));        
        w = w + ( Y(i) -  x );
    end
    
    w = w/N;

end