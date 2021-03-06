function [ error ] =classification_error_svm( Y, Y_dash )

    N = size(Y, 1);
    error = 0;
    for i = 1:N
        if Y(i) ~= Y_dash(i) 
            error = error + 1;
        end
    end
    error = error / N * 100;

end

