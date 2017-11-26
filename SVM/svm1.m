function [testY] = svm1(X,Y,testX)

    % we will here use one vs one technique 
    [N,D] = size(X);
    N
    C = 2 ;  % box constraint for soft margin
    gamma = 1.414;
    
    no_of_classes = 10 ;
    alphaMatrix= zeros(N,(no_of_classes*(no_of_classes-1))/2);
    
    W=zeros(1,(no_of_classes*(no_of_classes-1))/2);
    
    cur=0;
    for x = 1 : no_of_classes
        for y = x+1 : no_of_classes
            cur
            cur=cur+1;

            cnew=0;
            for s=1:N
                if Y(s,1)==x-1||Y(s,1)==y-1
                    cnew=cnew+1;
                end
            end

            Ycurrent=zeros(cnew,1);
            Xcurrent=zeros(cnew,D);

            index=1;
            for s=1:N
                if Y(s,1)==x-1||Y(s,1)==y-1
                    Xcurrent(index,:) = X(s,:);
                    if Y(s,1)==x-1
                        Ycurrent(index,1)=1;
                    else
                        Ycurrent(index,1)=-1; 
                    end
                    index=index+1;
                end
            end
      % using one vs one approach we will learn 10*(10-1)/2=45 different w and alpha
      
       % for class i the data belongs to  class x has value 1 else -1 for class y 
       'calculating kernel'
       Kernel = return_Kernel(Xcurrent,gamma);
       'alpha matrix calculating'   
       alphaMatrix(1:cnew,cur) = mysvmdualFormulation(Xcurrent,Ycurrent,Kernel,C); % lagrangian coefficient 
       'alpha matrix calculated'
       W(cur) = returnW(alphaMatrix(1:cnew,cur), Kernel, Ycurrent);
       
        end
    end
   
    
   'testing started' 
 Ntest=size(testX,1);
 testY=zeros(Ntest,1);
 cur=0;
 
 OutputMatrix=zeros(Ntest,no_of_classes);
 
 for p = 1 : no_of_classes
for q = p+1 : no_of_classes
cur
    cur=cur+1;    
cnew=0;
            for s=1:N
                if Y(s,1)==p-1||Y(s,1)==q-1
                    cnew=cnew+1;
                end
            end

            Ycurrent=zeros(cnew,1);
            Xcurrent=zeros(cnew,D);
index=1;
       for s=1:N
            if Y(s,1)==p-1||Y(s,1)==q-1
                    Xcurrent(index,:) = X(s,:);
                    if Y(s,1)==p-1
                        Ycurrent(index,1)=1;
                    else
                        Ycurrent(index,1)=-1; 
                    end
                    index=index+1;
            end
       end
       
       for i = 1:Ntest
           x = 0;
           for j = 1:cnew
               
               
               x = x + alphaMatrix(j,cur) * Ycurrent(j,1) * (gamma *( testX(i, :)*transpose( Xcurrent(j, :)))).^4;
               
               
           end
           
           testY(i) = W(cur) + x;
           
           if testY(i)>=0
               OutputMatrix(i,p)=OutputMatrix(i,p)+1;
           else
               OutputMatrix(i,q)=OutputMatrix(i,q)+1;
           end
     
      end
     
end
 end
 
 for i=1:Ntest
 [val, index]=max(OutputMatrix(i,:));
 testY(i,1)=index-1;
 end
 
 'test ended'
end

