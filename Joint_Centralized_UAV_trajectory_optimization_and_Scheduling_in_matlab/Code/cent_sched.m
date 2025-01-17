%% sample sched casadi

function y = cent_sched_casadi(pref,user_profile)

import casadi.*

%link_pref = abs(randn(2,3));
%link_pref = [1,2,3;2.2,1.4,0.8];

[M,K] = size(pref);

pref = pref.*user_profile;

%Part 1
alpha = SX.sym('alpha',M,K); %quad 1 scheduling

%Part 2

obj = -sum(sum(pref.*alpha));

%Part 3
g = [];
for i = 1:M
   g = [g;ones(1,K)*alpha(i,:)'];
end
for j = 1:K
   g = [g;ones(1,M)*alpha(:,j)];
end
         
P = [];  
 
% Part 4
OPT_variables = reshape(alpha,K*M,1);  % decision variable

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);


opts = struct;
opts.ipopt.max_iter = 1000;
opts.ipopt.print_level = 0; 
opts.print_time = 0; 
opts.ipopt.acceptable_tol =1e-8; 
opts.ipopt.acceptable_obj_change_tol = 1e-6; 

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

%Part 5
args = struct;
args.lbx = 0;  
args.ubx = 1;   
args.lbg = 0;  
args.ubg = 1;   

args.p   = []; 
args.x0  = 0.5; 

sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
    'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);

x_sol = full(sol.x)  ;          
min_value = -full(sol.f) ;

y = reshape(x_sol,M,K);

end








