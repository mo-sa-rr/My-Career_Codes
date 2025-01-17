function [y,x] = cent_pos(sched,user_loc,user_profile,UAV_loc,sigma2,H,max_speed,min_dist)




import casadi.*

init_sched = sched.*user_profile;
          
loc = user_loc;     % initial location of users

[~,K] = size(loc);  % number of users    

init_Q = UAV_loc;

[~,M] = size(init_Q);  % number of Uavs




% A = zeros(1,M*K);
% B = zeros(1,M*K);
% 
% 
% 
% for i = 1:M
%     for j = 1:K
%         temp = 0;
%         for l = 1:M
%             temp = temp + 1/(H^2+norm(init_Q(:,l)-loc(:,j))^2);
%         end
%         A(K*(i-1)+j) = log2(exp(1))/(H^2+norm(init_Q(:,i)-loc(:,j))^2)^2/(temp+sigma2);
%         B(K*(i-1)+j) = log2(temp+sigma2) ; 
%     end
% end



            

Q = SX.sym('Q',2,M);

Slack = SX.sym('Slack',1,K*M);

% Rlb = SX.zeros(1,K*M);
% 
% 
% for i = 1:M
%     for j = 1:K
%         
%         for k = 1:M            
%             Rlb(K*(i-1)+j) = Rlb(K*(i-1)+j) - A(K*(i-1)+j)*(norm(Q(:,k)-loc(:,j))^2-norm(init_Q(:,k)-loc(:,j))^2)+B(K*(i-1)+j);
%         end
%     end
% end



obj = SX.zeros(1,1);

% for i = 1:M
%     for j = 1:K
%     obj = obj + init_sched(i,j)*(Rlb(K*(i-1)+j)-log(sum(1/(H^2+Slack(j:K:end)))-1/(H^2+Slack(K*(i-1)+j))+sigma2)/log(2)  );
%     end
% end


for i = 1:M
    for j = 1:K
    %obj = obj + init_sched(i,j)*log(1+1/(H^2+norm(Q(:,i)-loc(:,j))^2)/(sum(1/(H^2+Slack(j:K:end)))-1/(H^2+Slack(K*(i-1)+j))+sigma2))/log(2) ;
    %obj = obj + init_sched(i,j)*log(1+1/(H^2+norm(Q(:,i)-loc(:,j))^2))/log(2) ;
    obj = obj + init_sched(i,j)*log(1+1/(H^2+norm(Q(:,i)-loc(:,j))^2)/(sum(1/(H^2+sum((Q-loc(:,j)).^2)))-1/(H^2+norm(Q(:,i)-loc(:,j))^2)+sigma2))/log(2) ;

    end
end

obj = -obj;

%////////////////Part 3 constraints

 g = [];
 for i = 1:M
%     for j = 1:K
%        g = [g; Slack(K*(i-1)+j)-norm(init_Q(:,i)-loc(:,j))^2-2*(init_Q(:,i)-loc(:,j))'*(Q(:,i)-init_Q(:,i))];
%     end
     g = [g; norm(Q(:,i)-init_Q(:,i))^2-max_speed^2];
 end


for i = 1:M
    for k = 1:M
     
        if i~=k
        %g = [g; min_dist^2 + norm(init_Q(:,i)-init_Q(:,k))^2-2*(init_Q(:,i)-init_Q(:,k))'*(Q(:,i)-Q(:,k))];
        g = [g; min_dist^2 - norm(Q(:,i)-Q(:,k))^2];
        end
        
    end

end

 
P = [];  
 
% Part 4
OPT_variables = [reshape(Q,2*M,1);Slack'];  %Two decision variable

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);


opts = struct;
opts.ipopt.max_iter = 1000;
opts.ipopt.print_level = 0; 
opts.print_time = 0; 
opts.ipopt.acceptable_tol =1e-8; 
opts.ipopt.acceptable_obj_change_tol = 1e-6;
%opts.monitor = char('nlp_g');


solver = nlpsol('solver', 'ipopt', nlp_prob,opts);



%Part 5
xlb = [-inf*ones(2*M,1);zeros(K*M,1)];

args = struct;
args.lbx = xlb;  
args.ubx = inf;   
args.lbg = -inf;  
args.ubg = 0;   

args.p   = []; 
args.x0  = 0.5; 

sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
    'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);

x_sol = full(sol.x)  ;          
min_value = -full(sol.f) ;


y = reshape(x_sol(1:2*M),2,M);
x = min_value;



end