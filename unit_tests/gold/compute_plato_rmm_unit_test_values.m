clear all
close all
clc
%% test settings
property_scale = 1.0; % set this to 1.0 for all tests except 3D_Expression_WithInertia_Density0p5 then set to 1.5
mesh_size = 1; %set this to match cMeshWidth in unit test

%% material properties
% Ce
mu_e = property_scale*557.11;
lam_e = -property_scale*120.74;
mus_e = property_scale*8.37;

C_e = [lam_e + 2*mu_e, lam_e, lam_e, 0, 0, 0;
    lam_e, lam_e + 2*mu_e, lam_e, 0, 0, 0;
    lam_e, lam_e, lam_e + 2*mu_e, 0, 0, 0;
    0, 0, 0, mus_e, 0, 0;
    0, 0, 0, 0, mus_e, 0;
    0, 0, 0, 0, 0, mus_e];

% Cc
mu_c = property_scale*1.8e-4;
C_c = mu_c*diag(ones(3,1));

% Cm
mu_m = property_scale*255.71;
lam_m = property_scale*180.63;
mus_m = property_scale*181.28;

C_m = [lam_m + 2*mu_m, lam_m, lam_m, 0, 0, 0;
    lam_m, lam_m + 2*mu_m, lam_m, 0, 0, 0;
    lam_m, lam_m, lam_m + 2*mu_m, 0, 0, 0;
    0, 0, 0, mus_m, 0, 0;
    0, 0, 0, 0, mus_m, 0;
    0, 0, 0, 0, 0, mus_m];

%% inertia properties
% macroscopic mass density
rho = 1451.8;

% Te
eta_bar_1 = property_scale*0.6;
eta_bar_3 = property_scale*2.0;
eta_bar_star_1 = property_scale*0.2;

T_e = [eta_bar_3 + 2*eta_bar_1, eta_bar_3, eta_bar_3, 0, 0, 0;
    eta_bar_3, eta_bar_3 + 2*eta_bar_1, eta_bar_3, 0, 0, 0;
    eta_bar_3, eta_bar_3, eta_bar_3 + 2*eta_bar_1, 0, 0, 0;
    0, 0, 0, eta_bar_star_1, 0, 0;
    0, 0, 0, 0, eta_bar_star_1, 0;
    0, 0, 0, 0, 0, eta_bar_star_1];

% Tc
eta_bar_2 = property_scale*1.0e-4;
T_c = eta_bar_2*diag(ones(3,1));

% Jm
eta_1 = property_scale*2300.0;
eta_3 = -property_scale*1800.0;
eta_star_1 = property_scale*4500;

J_m = [eta_3 + 2*eta_1, eta_3, eta_3, 0, 0, 0;
    eta_3, eta_3 + 2*eta_1, eta_3, 0, 0, 0;
    eta_3, eta_3, eta_3 + 2*eta_1, 0, 0, 0;
    0, 0, 0, eta_star_1, 0, 0;
    0, 0, 0, 0, eta_star_1, 0;
    0, 0, 0, 0, 0, eta_star_1];

% Jc
eta_2 = property_scale*1.0e-4;
J_c = eta_2*diag(ones(3,1));

% % macroscopic mass density
% rho = 0;
% 
% % Te
% eta_bar_1 = 0;
% eta_bar_3 = 0;
% eta_bar_star_1 = 0;
% 
% T_e = [eta_bar_3 + 2*eta_bar_1, eta_bar_3, eta_bar_3, 0, 0, 0;
%     eta_bar_3, eta_bar_3 + 2*eta_bar_1, eta_bar_3, 0, 0, 0;
%     eta_bar_3, eta_bar_3, eta_bar_3 + 2*eta_bar_1, 0, 0, 0;
%     0, 0, 0, eta_bar_star_1, 0, 0;
%     0, 0, 0, 0, eta_bar_star_1, 0;
%     0, 0, 0, 0, 0, eta_bar_star_1];
% 
% % Tc
% eta_bar_2 = 0;
% T_c = eta_bar_2*diag(ones(3,1));
% 
% % Jm
% eta_1 = 0;
% eta_3 = 0;
% eta_star_1 = 0;
% 
% J_m = [eta_3 + 2*eta_1, eta_3, eta_3, 0, 0, 0;
%     eta_3, eta_3 + 2*eta_1, eta_3, 0, 0, 0;
%     eta_3, eta_3, eta_3 + 2*eta_1, 0, 0, 0;
%     0, 0, 0, eta_star_1, 0, 0;
%     0, 0, 0, 0, eta_star_1, 0;
%     0, 0, 0, 0, 0, eta_star_1];
% 
% % Jc
% eta_2 = 0;
% J_c = eta_2*diag(ones(3,1));

%% Element Data
num_dims = 3; % number of space dimentions
nele = 6; % only checking the first 6 elements
nodes_per_ele = 4; % tet4

% Connectivity
conn = zeros(nele,nodes_per_ele);
if mesh_width == 1
    conn(1,:) = [0,12,3,13] + 1; % original mesh (mesh width 2)
    conn(2,:) = [0,3,4,13] + 1;
    conn(3,:) = [0,4,1,13] + 1;
    conn(4,:) = [0,1,10,13] + 1;
    conn(5,:) = [0,10,9,13] + 1;
    conn(6,:) = [0,9,12,13] + 1;
elseif mesh_width == 2
    conn(1,:) = [0,6,2,7] + 1; % 6 ele mesh (mesh width 1)
    conn(2,:) = [0,2,3,7] + 1;
    conn(3,:) = [0,3,1,7] + 1;
    conn(4,:) = [0,1,5,7] + 1;
    conn(5,:) = [0,5,4,7] + 1;
    conn(6,:) = [0,4,6,7] + 1;
end

% Gauss Points / Weights

% 1-point:
% gp = zeros(1,3);
% gp(1,:) = [1.0/4.0, 1.0/4.0, 1.0/4.0];
% gw = [1.0/6.0];

% 4-point:
gp = zeros(4,3);
gp(1,:) = [0.585410196624969, 0.138196601125011, 0.138196601125011];
gp(2,:) = [0.138196601125011, 0.585410196624969, 0.138196601125011];
gp(3,:) = [0.138196601125011, 0.138196601125011, 0.585410196624969];
gp(4,:) = [0.138196601125011, 0.138196601125011, 0.138196601125011];
gw = [1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0];

% Shape functions
num_gp = length(gw);
all_shape_functions = zeros(num_gp,nodes_per_ele);
for i = 1:num_gp
    all_shape_functions(i,:) = [1.0-gp(i,1)-gp(i,2)-gp(i,3), gp(i,1), gp(i,2), gp(i,3)];
end

% Shape functions for first gauss point
shape_functions = all_shape_functions(1,:);

% Volume jacobian times quadrature weight
cell_weight = 0.0208333333333333; 
cell_volume = 0.125;

% Shape function gradients
shape_derivs = zeros(nele,nodes_per_ele,3);

shape_derivs(1,1,:) = [0.0, -2.0, 0.0];
shape_derivs(1,2,:) = [2.0, 0.0,-2.0];
shape_derivs(1,3,:) = [-2.0, 2.0, 0.0];
shape_derivs(1,4,:) = [0.0, 0.0, 2.0];

shape_derivs(2,1,:) = [0.0, -2.0, 0.0];
shape_derivs(2,2,:) = [0.0, 2.0, -2.0];
shape_derivs(2,3,:) = [-2.0, 0.0, 2.0];
shape_derivs(2,4,:) = [2.0, 0.0, 0.0];

shape_derivs(3,1,:) = [0.0, 0.0, -2.0];
shape_derivs(3,2,:) = [-2.0, 2.0, 0.0];
shape_derivs(3,3,:) = [0.0, -2.0, 2.0];
shape_derivs(3,4,:) = [2.0, 0.0, 0.0];

shape_derivs(4,1,:) = [0.0, 0.0, -2.0];
shape_derivs(4,2,:) = [-2.0, 0.0, 2.0];
shape_derivs(4,3,:) = [2.0, -2.0, 0.0];
shape_derivs(4,4,:) = [0.0, 2.0, 0.0];

shape_derivs(5,1,:) = [-2.0, 0.0, 0.0];
shape_derivs(5,2,:) = [0.0, -2.0, 2.0];
shape_derivs(5,3,:) = [2.0, 0.0, -2.0];
shape_derivs(5,4,:) = [0.0, 2.0, 0.0];

shape_derivs(6,1,:) = [-2.0, 0.0, 0.0];
shape_derivs(6,2,:) = [2.0, -2.0, 0.0];
shape_derivs(6,3,:) = [0.0, 2.0, -2.0];
shape_derivs(6,4,:) = [0.0, 0.0, 2.0];

%% Displacement and acceleration
max_node = 14; % Max node ID for recording (mesh has 27 total nodes)

disp_scale = 1.0e-7;
acc_scale = 1.0e-5;

disp_node_vals = zeros(max_node,3);
acc_node_vals = zeros(max_node,3);

chi_node_vals = zeros(max_node,9);
chidd_node_vals = zeros(max_node,9);

for i = 1:max_node
    disp_node_vals(i,:) = disp_scale*(i - 1)*[1,2,3];
    acc_node_vals(i,:) = acc_scale*(i - 1)*[1,2,3];
    
    chi_node_vals(i,:) = disp_scale*(i - 1)*[4,5,6,7,8,9,10,11,12];  % stored as [11, 22, 33, 23, 13, 12, 32, 31, 21]
    chidd_node_vals(i,:) = acc_scale*(i - 1)*[4,5,6,7,8,9,10,11,12]; % stored as [11, 22, 33, 23, 13, 12, 32, 31, 21]
end

%% Element State Strains
ele_sym_ugrad = zeros(nele,num_gp,6);
ele_skw_ugrad = zeros(nele,num_gp,3);
ele_sym_chi = zeros(nele,num_gp,6);
ele_skw_chi = zeros(nele,num_gp,3);
for i = 1:nele
    for j = 1:num_gp
        ele_disp_vals = disp_node_vals(conn(i,:),:);
        ele_chi_vals = chi_node_vals(conn(i,:),:);

        ele_sym_ugrad(i,j,1) = shape_derivs(i,:,1)*ele_disp_vals(:,1);
        ele_sym_ugrad(i,j,2) = shape_derivs(i,:,2)*ele_disp_vals(:,2);
        ele_sym_ugrad(i,j,3) = shape_derivs(i,:,3)*ele_disp_vals(:,3);
        ele_sym_ugrad(i,j,4) = shape_derivs(i,:,3)*ele_disp_vals(:,2) + shape_derivs(i,:,2)*ele_disp_vals(:,3);
        ele_sym_ugrad(i,j,5) = shape_derivs(i,:,3)*ele_disp_vals(:,1) + shape_derivs(i,:,1)*ele_disp_vals(:,3);
        ele_sym_ugrad(i,j,6) = shape_derivs(i,:,2)*ele_disp_vals(:,1) + shape_derivs(i,:,1)*ele_disp_vals(:,2);

        ele_skw_ugrad(i,j,1) = shape_derivs(i,:,3)*ele_disp_vals(:,2) - shape_derivs(i,:,2)*ele_disp_vals(:,3);
        ele_skw_ugrad(i,j,2) = shape_derivs(i,:,3)*ele_disp_vals(:,1) - shape_derivs(i,:,1)*ele_disp_vals(:,3);
        ele_skw_ugrad(i,j,3) = shape_derivs(i,:,2)*ele_disp_vals(:,1) - shape_derivs(i,:,1)*ele_disp_vals(:,2);

        ele_sym_chi(i,j,1) = all_shape_functions(j,:)*ele_chi_vals(:,1);
        ele_sym_chi(i,j,2) = all_shape_functions(j,:)*ele_chi_vals(:,2);
        ele_sym_chi(i,j,3) = all_shape_functions(j,:)*ele_chi_vals(:,3);
        ele_sym_chi(i,j,4) = all_shape_functions(j,:)*ele_chi_vals(:,4) + all_shape_functions(j,:)*ele_chi_vals(:,7);
        ele_sym_chi(i,j,5) = all_shape_functions(j,:)*ele_chi_vals(:,5) + all_shape_functions(j,:)*ele_chi_vals(:,8);
        ele_sym_chi(i,j,6) = all_shape_functions(j,:)*ele_chi_vals(:,6) + all_shape_functions(j,:)*ele_chi_vals(:,9);

        ele_skw_chi(i,j,1) = all_shape_functions(j,:)*ele_chi_vals(:,4) - all_shape_functions(j,:)*ele_chi_vals(:,7);
        ele_skw_chi(i,j,2) = all_shape_functions(j,:)*ele_chi_vals(:,5) - all_shape_functions(j,:)*ele_chi_vals(:,8);
        ele_skw_chi(i,j,3) = all_shape_functions(j,:)*ele_chi_vals(:,6) - all_shape_functions(j,:)*ele_chi_vals(:,9);
    end
end

%% Element Inertia Strains
ele_sym_agrad = zeros(nele,num_gp,6);
ele_skw_agrad = zeros(nele,num_gp,3);
ele_sym_chidd = zeros(nele,num_gp,6);
ele_skw_chidd = zeros(nele,num_gp,3);
for i = 1:nele
    for j = 1:num_gp
        ele_acc_vals = acc_node_vals(conn(i,:),:);
        ele_chidd_vals = chidd_node_vals(conn(i,:),:);

        ele_sym_agrad(i,j,1) = shape_derivs(i,:,1)*ele_acc_vals(:,1);
        ele_sym_agrad(i,j,2) = shape_derivs(i,:,2)*ele_acc_vals(:,2);
        ele_sym_agrad(i,j,3) = shape_derivs(i,:,3)*ele_acc_vals(:,3);
        ele_sym_agrad(i,j,4) = shape_derivs(i,:,3)*ele_acc_vals(:,2) + shape_derivs(i,:,2)*ele_acc_vals(:,3);
        ele_sym_agrad(i,j,5) = shape_derivs(i,:,3)*ele_acc_vals(:,1) + shape_derivs(i,:,1)*ele_acc_vals(:,3);
        ele_sym_agrad(i,j,6) = shape_derivs(i,:,2)*ele_acc_vals(:,1) + shape_derivs(i,:,1)*ele_acc_vals(:,2);

        ele_skw_agrad(i,j,1) = shape_derivs(i,:,3)*ele_acc_vals(:,2) - shape_derivs(i,:,2)*ele_acc_vals(:,3);
        ele_skw_agrad(i,j,2) = shape_derivs(i,:,3)*ele_acc_vals(:,1) - shape_derivs(i,:,1)*ele_acc_vals(:,3);
        ele_skw_agrad(i,j,3) = shape_derivs(i,:,2)*ele_acc_vals(:,1) - shape_derivs(i,:,1)*ele_acc_vals(:,2);

        ele_sym_chidd(i,j,1) = all_shape_functions(j,:)*ele_chidd_vals(:,1);
        ele_sym_chidd(i,j,2) = all_shape_functions(j,:)*ele_chidd_vals(:,2);
        ele_sym_chidd(i,j,3) = all_shape_functions(j,:)*ele_chidd_vals(:,3);
        ele_sym_chidd(i,j,4) = all_shape_functions(j,:)*ele_chidd_vals(:,4) + all_shape_functions(j,:)*ele_chidd_vals(:,7);
        ele_sym_chidd(i,j,5) = all_shape_functions(j,:)*ele_chidd_vals(:,5) + all_shape_functions(j,:)*ele_chidd_vals(:,8);
        ele_sym_chidd(i,j,6) = all_shape_functions(j,:)*ele_chidd_vals(:,6) + all_shape_functions(j,:)*ele_chidd_vals(:,9);

        ele_skw_chidd(i,j,1) = all_shape_functions(j,:)*ele_chidd_vals(:,4) - all_shape_functions(j,:)*ele_chidd_vals(:,7);
        ele_skw_chidd(i,j,2) = all_shape_functions(j,:)*ele_chidd_vals(:,5) - all_shape_functions(j,:)*ele_chidd_vals(:,8);
        ele_skw_chidd(i,j,3) = all_shape_functions(j,:)*ele_chidd_vals(:,6) - all_shape_functions(j,:)*ele_chidd_vals(:,9);
    end
end

%% compute state stresses
ele_sym_cauchy = zeros(nele,num_gp,6);
ele_skw_cauchy = zeros(nele,num_gp,6);
ele_sym_s = zeros(nele,num_gp,6);

for i = 1:nele
    for j = 1:num_gp
        sym_ugrad_minus_sym_chi = squeeze(ele_sym_ugrad(i,j,:) - ele_sym_chi(i,j,:));
        ele_sym_cauchy(i,j,:) = C_e*sym_ugrad_minus_sym_chi;

        skw_ugrad_minus_skw_chi = squeeze(ele_skw_ugrad(i,j,:) - ele_skw_chi(i,j,:));
        ele_skw_cauchy(i,j,4:end) = C_c*skw_ugrad_minus_skw_chi;

        gp_sym_chi = squeeze(ele_sym_chi(i,j,:));
        ele_sym_s(i,j,:) = C_m*gp_sym_chi;
    end
end

%% Compute inertia Stresses
ele_sym_meso = zeros(nele,num_gp,6);
ele_skw_meso = zeros(nele,num_gp,6);
ele_sym_micro = zeros(nele,num_gp,6);
ele_skw_micro = zeros(nele,num_gp,6);

for i = 1:nele
    for j = 1:num_gp
        ele_sym_meso(i,j,:) = T_e*squeeze(ele_sym_agrad(i,j,:));
        ele_skw_meso(i,j,4:end) = T_c*squeeze(ele_skw_agrad(i,j,:));
        ele_sym_micro(i,j,:) = J_m*squeeze(ele_sym_chidd(i,j,:));
        ele_skw_micro(i,j,4:end) = J_c*squeeze(ele_skw_chidd(i,j,:));
    end
end

%% Compute weighting (ComputeLinearInertiaKinetics)
control_ws = zeros(nele,nodes_per_ele);
control_ws(1,:) = [0.0, 1.0, 0.5, 0.5];
control_ws(2,:) = [0.8, 1.0, 0.5, 0.7];
control_ws(3,:) = [0.4, 0.05, 0.3, 0.25];
ele_weight = zeros(1,nele);
for i = 1:nele
    for j = 1:nodes_per_ele
        ele_weight(i) = ele_weight(i) + shape_functions(j)*control_ws(i,j);
    end
end

%% Compute weighting (ComputeExpressionInertiaKinetics)
control_ws = zeros(nele,nodes_per_ele);
control_ws(1,:) = [0.8, 1.0, 0.5, 0.7];
control_ws(2,:) = [0.2, 0.4, 0.6, 0.8];
control_ws(3,:) = [0.4, 0.05, 0.3, 0.25];
ele_weight = zeros(1,nele);
for i = 1:nele
    for j = 1:nodes_per_ele
        ele_weight(i) = ele_weight(i) + shape_functions(j)*control_ws(i,j);
    end
end

%% Compute Stress Divergences
ele_sym_divergence = zeros(nele,nodes_per_ele,3);
ele_skw_divergence = zeros(nele,nodes_per_ele,3);
ele_full_divergence = zeros(nele,nodes_per_ele,3);

for iEle = 1:nele
    for iGp = 1:num_gp
        for iNode = 1:nodes_per_ele
            ele_sym_divergence(iEle,iNode,1) = ele_sym_divergence(iEle,iNode,1) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*ele_sym_cauchy(iEle,iGp,1) + ...
                shape_derivs(iEle,iNode,2)*ele_sym_cauchy(iEle,iGp,6) + shape_derivs(iEle,iNode,3)*ele_sym_cauchy(iEle,iGp,5));
            ele_sym_divergence(iEle,iNode,2) = ele_sym_divergence(iEle,iNode,2) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*ele_sym_cauchy(iEle,iGp,6) + ...
                shape_derivs(iEle,iNode,2)*ele_sym_cauchy(iEle,iGp,2) + shape_derivs(iEle,iNode,3)*ele_sym_cauchy(iEle,iGp,4));
            ele_sym_divergence(iEle,iNode,3) = ele_sym_divergence(iEle,iNode,3) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*ele_sym_cauchy(iEle,iGp,5) + ...
                shape_derivs(iEle,iNode,2)*ele_sym_cauchy(iEle,iGp,4) + shape_derivs(iEle,iNode,3)*ele_sym_cauchy(iEle,iGp,3));

            ele_skw_divergence(iEle,iNode,1) = ele_skw_divergence(iEle,iNode,1) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,2)*ele_skw_cauchy(iEle,iGp,6) + shape_derivs(iEle,iNode,3)*ele_skw_cauchy(iEle,iGp,5));
            ele_skw_divergence(iEle,iNode,2) = ele_skw_divergence(iEle,iNode,2) + gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*ele_skw_cauchy(iEle,iGp,6) + shape_derivs(iEle,iNode,3)*ele_skw_cauchy(iEle,iGp,4));
            ele_skw_divergence(iEle,iNode,3) = ele_skw_divergence(iEle,iNode,3) + gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*ele_skw_cauchy(iEle,iGp,5) - shape_derivs(iEle,iNode,2)*ele_skw_cauchy(iEle,iGp,4));
        end
    end
    ele_full_divergence(iEle,:,:) = ele_sym_divergence(iEle,:,:) + ele_skw_divergence(iEle,:,:);
end

%% Compute Stress Values At Nodes
ele_nodal_sym_s = zeros(nele,nodes_per_ele,9);
ele_nodal_sym_cauchy = zeros(nele,nodes_per_ele,9);
ele_nodal_skw_cauchy = zeros(nele,nodes_per_ele,9);
ele_full_nodal_stress = zeros(nele,nodes_per_ele,9);

for iEle = 1:nele
    for iGp = 1:num_gp
        for iNode = 1:nodes_per_ele
            ele_nodal_sym_s(iEle,iNode,1:3) = ele_nodal_sym_s(iEle,iNode,1:3) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_s(iEle,iGp,1:3);
            ele_nodal_sym_s(iEle,iNode,4:6) = ele_nodal_sym_s(iEle,iNode,4:6) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_s(iEle,iGp,4:6);
            ele_nodal_sym_s(iEle,iNode,7:9) = ele_nodal_sym_s(iEle,iNode,7:9) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_s(iEle,iGp,4:6);

            ele_nodal_sym_cauchy(iEle,iNode,1:3) = ele_nodal_sym_cauchy(iEle,iNode,1:3) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_cauchy(iEle,iGp,1:3);
            ele_nodal_sym_cauchy(iEle,iNode,4:6) = ele_nodal_sym_cauchy(iEle,iNode,4:6) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_cauchy(iEle,iGp,4:6);
            ele_nodal_sym_cauchy(iEle,iNode,7:9) = ele_nodal_sym_cauchy(iEle,iNode,7:9) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_cauchy(iEle,iGp,4:6);

            ele_nodal_skw_cauchy(iEle,iNode,1:3) = ele_nodal_skw_cauchy(iEle,iNode,1:3) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_skw_cauchy(iEle,iGp,1:3);
            ele_nodal_skw_cauchy(iEle,iNode,4:6) = ele_nodal_skw_cauchy(iEle,iNode,4:6) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_skw_cauchy(iEle,iGp,4:6);
            ele_nodal_skw_cauchy(iEle,iNode,7:9) = ele_nodal_skw_cauchy(iEle,iNode,7:9) - gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_skw_cauchy(iEle,iGp,4:6);
        end
    end
    ele_full_nodal_stress(iEle,:,:) = ele_nodal_sym_s(iEle,:,:) - ele_nodal_sym_cauchy(iEle,:,:) - ele_nodal_skw_cauchy(iEle,:,:);
end

%% compute inertia stress divergences
ele_sym_inertia_divergence = zeros(nele,nodes_per_ele,12);
ele_skw_inertia_divergence = zeros(nele,nodes_per_ele,12);
ele_full_inertia_divergence = zeros(nele,nodes_per_ele,12);

for iEle = 1:nele
    for iGp = 1:num_gp
        for iNode = 1:nodes_per_ele
            ele_sym_inertia_divergence(iEle,iNode,1) = ele_sym_inertia_divergence(iEle,iNode,1) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*ele_sym_meso(iEle,iGp,1) + ...
                shape_derivs(iEle,iNode,2)*ele_sym_meso(iEle,iGp,6) + shape_derivs(iEle,iNode,3)*ele_sym_meso(iEle,iGp,5));
            ele_sym_inertia_divergence(iEle,iNode,2) = ele_sym_inertia_divergence(iEle,iNode,2) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*ele_sym_meso(iEle,iGp,6) + ...
                shape_derivs(iEle,iNode,2)*ele_sym_meso(iEle,iGp,2) + shape_derivs(iEle,iNode,3)*ele_sym_meso(iEle,iGp,4));
            ele_sym_inertia_divergence(iEle,iNode,3) = ele_sym_inertia_divergence(iEle,iNode,3) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*ele_sym_meso(iEle,iGp,5) + ...
                shape_derivs(iEle,iNode,2)*ele_sym_meso(iEle,iGp,4) + shape_derivs(iEle,iNode,3)*ele_sym_meso(iEle,iGp,3));

            ele_skw_inertia_divergence(iEle,iNode,1) = ele_skw_inertia_divergence(iEle,iNode,1) + gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,2)*ele_skw_meso(iEle,iGp,6) + shape_derivs(iEle,iNode,3)*ele_skw_meso(iEle,iGp,5));
            ele_skw_inertia_divergence(iEle,iNode,2) = ele_skw_inertia_divergence(iEle,iNode,2) + gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*ele_skw_meso(iEle,iGp,6) + shape_derivs(iEle,iNode,3)*ele_skw_meso(iEle,iGp,4));
            ele_skw_inertia_divergence(iEle,iNode,3) = ele_skw_inertia_divergence(iEle,iNode,3) + gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*ele_skw_meso(iEle,iGp,5) - shape_derivs(iEle,iNode,2)*ele_skw_meso(iEle,iGp,4));
        end
    end
    ele_full_inertia_divergence(iEle,:,:) = ele_sym_inertia_divergence(iEle,:,:) + ele_skw_inertia_divergence(iEle,:,:);
end

%% compute inertia stress values at nodes
ele_nodal_sym_micro = zeros(nele,nodes_per_ele,9);
ele_nodal_skw_micro = zeros(nele,nodes_per_ele,9);
ele_full_nodal_inertia_stress = zeros(nele,nodes_per_ele,9);

for iEle = 1:nele
    for iGp = 1:num_gp
        for iNode = 1:nodes_per_ele
            ele_nodal_sym_micro(iEle,iNode,1:3) = ele_nodal_sym_micro(iEle,iNode,1:3) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_micro(iEle,iGp,1:3);
            ele_nodal_sym_micro(iEle,iNode,4:6) = ele_nodal_sym_micro(iEle,iNode,4:6) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_micro(iEle,iGp,4:6);
            ele_nodal_sym_micro(iEle,iNode,7:9) = ele_nodal_sym_micro(iEle,iNode,7:9) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_sym_micro(iEle,iGp,4:6);

            ele_nodal_skw_micro(iEle,iNode,1:3) = ele_nodal_skw_micro(iEle,iNode,1:3) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_skw_micro(iEle,iGp,1:3);
            ele_nodal_skw_micro(iEle,iNode,4:6) = ele_nodal_skw_micro(iEle,iNode,4:6) + gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_skw_micro(iEle,iGp,4:6);
            ele_nodal_skw_micro(iEle,iNode,7:9) = ele_nodal_skw_micro(iEle,iNode,7:9) - gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*ele_skw_micro(iEle,iGp,4:6);
        end
    end
    ele_full_nodal_inertia_stress(iEle,:,:) = ele_nodal_sym_micro(iEle,:,:) + ele_nodal_skw_micro(iEle,:,:);
end

%% Compute macro inertia at nodes
ele_node_inertia = zeros(nele,nodes_per_ele,3);
ele_total_inertia_residual = zeros(nele,nodes_per_ele,3);

for iEle = 1:nele
    ele_acc_vals = acc_node_vals(conn(iEle,:),:);
    for iGp = 1:length(gw)
        ele_inertia_gp = rho*all_shape_functions(iGp,:)*ele_acc_vals;
        for iNode = 1:nodes_per_ele
            for iDim = 1:num_dims
                ele_node_inertia(iEle,iNode,iDim) = ele_node_inertia(iEle,iNode,iDim) + cell_volume*gw(iGp)*all_shape_functions(iGp,iNode)*ele_inertia_gp(iDim);
            end
        end
    end
    ele_total_inertia_residual(iEle,:,:) = ele_node_inertia(iEle,:,:) + ele_full_inertia_divergence(iEle,:,1:3);
end

%% Full element residuals
ele_stress_residual = zeros(nele,nodes_per_ele,12);
ele_inertia_residual = zeros(nele,nodes_per_ele,12);
for iEle = 1:nele
    ele_stress_residual(iEle,:,1:3) = ele_full_divergence(iEle,:,:);
    ele_stress_residual(iEle,:,4:end) = ele_full_nodal_stress(iEle,:,:);
    ele_inertia_residual(iEle,:,1:3) = ele_total_inertia_residual(iEle,:,:);
    ele_inertia_residual(iEle,:,4:end) = ele_full_nodal_inertia_stress(iEle,:,:);
end

%% Assemble global residuals
node0_global_residual = zeros(12,1);
node13_global_residual = zeros(12,1);
for iEle = 1:nele
    local_contribution0 = ele_stress_residual(iEle,conn(iEle,:) == 1, :) + ele_inertia_residual(iEle,conn(iEle,:) == 1, :);
    node0_global_residual = node0_global_residual + local_contribution0(:);
    
    local_contribution13 = ele_stress_residual(iEle,conn(iEle,:) == 14, :) + ele_inertia_residual(iEle,conn(iEle,:) == 14, :);
    node13_global_residual = node13_global_residual + local_contribution13(:);
end

%% Compute Gradient w.r.t. U
dfull_divergence = zeros(nele,num_gp,12*nodes_per_ele,12*nodes_per_ele);
dfull_nodal_stress = zeros(nele,num_gp,12*nodes_per_ele,12*nodes_per_ele);
for iEle = 1:nele
    for iGp = 1:num_gp
        % strain derivatives
        dsym_ugrad = zeros(6,12*nodes_per_ele);
        dskw_ugrad = zeros(3,12*nodes_per_ele);
        dsym_chi = zeros(6,12*nodes_per_ele);
        dskw_chi = zeros(3,12*nodes_per_ele);

        offset = 0;
        for iNode = 1:nodes_per_ele
            dsym_ugrad(1,offset+1:offset+3) = [shape_derivs(iEle,iNode,1), 0, 0];
            dsym_ugrad(2,offset+1:offset+3) = [0, shape_derivs(iEle,iNode,2), 0];
            dsym_ugrad(3,offset+1:offset+3) = [0, 0, shape_derivs(iEle,iNode,3)];
            dsym_ugrad(4,offset+1:offset+3) = [0, shape_derivs(iEle,iNode,3), shape_derivs(iEle,iNode,2)];
            dsym_ugrad(5,offset+1:offset+3) = [shape_derivs(iEle,iNode,3), 0, shape_derivs(iEle,iNode,1)];
            dsym_ugrad(6,offset+1:offset+3) = [shape_derivs(iEle,iNode,2), shape_derivs(iEle,iNode,1), 0];

            dskw_ugrad(1,offset+1:offset+3) = [0, shape_derivs(iEle,iNode,3), -shape_derivs(iEle,iNode,2)];
            dskw_ugrad(2,offset+1:offset+3) = [shape_derivs(iEle,iNode,3), 0, -shape_derivs(iEle,iNode,1)];
            dskw_ugrad(3,offset+1:offset+3) = [shape_derivs(iEle,iNode,2), -shape_derivs(iEle,iNode,1), 0];

            dsym_chi(1,offset+4:offset+12) = [all_shape_functions(iGp,iNode), 0, 0, 0, 0, 0, 0, 0, 0];
            dsym_chi(2,offset+4:offset+12) = [0, all_shape_functions(iGp,iNode), 0, 0, 0, 0, 0, 0, 0];
            dsym_chi(3,offset+4:offset+12) = [0, 0, all_shape_functions(iGp,iNode), 0, 0, 0, 0, 0, 0];
            dsym_chi(4,offset+4:offset+12) = [0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, all_shape_functions(iGp,iNode), 0, 0];
            dsym_chi(5,offset+4:offset+12) = [0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, all_shape_functions(iGp,iNode), 0];
            dsym_chi(6,offset+4:offset+12) = [0, 0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, all_shape_functions(iGp,iNode)];

            dskw_chi(1,offset+4:offset+12) = [0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, -all_shape_functions(iGp,iNode), 0, 0];
            dskw_chi(2,offset+4:offset+12) = [0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, -all_shape_functions(iGp,iNode), 0];
            dskw_chi(3,offset+4:offset+12) = [0, 0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, -all_shape_functions(iGp,iNode)];

            offset = offset + 12;
        end

        % stress derivatives
        dsym_cauchy = C_e*(dsym_ugrad - dsym_chi);
        dskw_cauchy = zeros(6,12*nodes_per_ele);
        dskw_cauchy(4:end,:) = C_c*(dskw_ugrad - dskw_chi);
        dsym_s = C_m*dsym_chi;

        % stress divergence derivatives
        dsym_divergence = zeros(12*nodes_per_ele,12*nodes_per_ele);
        dskw_divergence = zeros(12*nodes_per_ele,12*nodes_per_ele);

        offset = 0;
        for iNode = 1:nodes_per_ele
            dsym_divergence(offset+1,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*dsym_cauchy(1,:) + ...
                shape_derivs(iEle,iNode,2)*dsym_cauchy(6,:) + shape_derivs(iEle,iNode,3)*dsym_cauchy(5,:));
            dsym_divergence(offset+2,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*dsym_cauchy(6,:) + ...
                shape_derivs(iEle,iNode,2)*dsym_cauchy(2,:) + shape_derivs(iEle,iNode,3)*dsym_cauchy(4,:));
            dsym_divergence(offset+3,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*dsym_cauchy(5,:) + ...
                shape_derivs(iEle,iNode,2)*dsym_cauchy(4,:) + shape_derivs(iEle,iNode,3)*dsym_cauchy(3,:));

            dskw_divergence(offset+1,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,2)*dskw_cauchy(6,:) + shape_derivs(iEle,iNode,3)*dskw_cauchy(5,:));
            dskw_divergence(offset+2,:) = gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*dskw_cauchy(6,:) + shape_derivs(iEle,iNode,3)*dskw_cauchy(4,:));
            dskw_divergence(offset+3,:) = gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*dskw_cauchy(5,:) - shape_derivs(iEle,iNode,2)*dskw_cauchy(4,:));

            offset = offset + 12;
        end
        dfull_divergence(iEle,iGp,:,:) = dsym_divergence + dskw_divergence;

        % stress values at nodes derivatives
        dnodal_sym_s = zeros(12*nodes_per_ele,12*nodes_per_ele);
        dnodal_sym_cauchy = zeros(12*nodes_per_ele,12*nodes_per_ele);
        dnodal_skw_cauchy = zeros(12*nodes_per_ele,12*nodes_per_ele);

        offset = 0;
        for iNode = 1:nodes_per_ele
            dnodal_sym_s(offset+4:offset+6,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_s(1:3,:);
            dnodal_sym_s(offset+7:offset+9,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_s(4:6,:);
            dnodal_sym_s(offset+10:offset+12,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_s(4:6,:);

            dnodal_sym_cauchy(offset+4:offset+6,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_cauchy(1:3,:);
            dnodal_sym_cauchy(offset+7:offset+9,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_cauchy(4:6,:);
            dnodal_sym_cauchy(offset+10:offset+12,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_cauchy(4:6,:);

            dnodal_skw_cauchy(offset+4:offset+6,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dskw_cauchy(1:3,:);
            dnodal_skw_cauchy(offset+7:offset+9,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dskw_cauchy(4:6,:);
            dnodal_skw_cauchy(offset+10:offset+12,:) = -gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dskw_cauchy(4:6,:);

            offset = offset + 12;
        end
        dfull_nodal_stress(iEle,iGp,:,:) = dnodal_sym_s - dnodal_sym_cauchy - dnodal_skw_cauchy;
    end
end

% Assembly
node0_global_jac = zeros(12,12);
for iEle = 1:nele
    for iGp = 1:num_gp
        local_conn = find(conn(iEle,:) == 1);
        idx = local_conn:12*local_conn;
        local_contribution = dfull_divergence(iEle, iGp, idx, idx) + dfull_nodal_stress(iEle, iGp, idx, idx);
        node0_global_jac = node0_global_jac + reshape(local_contribution,12,12);
    end
end

%% Compute Gradient w.r.t. A
dfull_nodal_inertia_stress = zeros(nele,iGp,12*nodes_per_ele,12*nodes_per_ele);
dtotal_inertia_residual = zeros(nele,iGp,12*nodes_per_ele,12*nodes_per_ele);
for iEle = 1:nele
    for iGp = 1:num_gp
        % inertia strain derivatives
        dsym_agrad = zeros(6,12*nodes_per_ele);
        dskw_agrad = zeros(3,12*nodes_per_ele);
        dsym_chidd = zeros(6,12*nodes_per_ele);
        dskw_chidd = zeros(3,12*nodes_per_ele);

        offset = 0;
        for iNode = 1:nodes_per_ele
            ele_acc_vals = acc_node_vals(conn(i,:),:);
            ele_chidd_vals = chidd_node_vals(conn(i,:),:);

            dsym_agrad(1,offset+1:offset+3) = [shape_derivs(iEle,iNode,1), 0, 0];
            dsym_agrad(2,offset+1:offset+3) = [0, shape_derivs(iEle,iNode,2), 0];
            dsym_agrad(3,offset+1:offset+3) = [0, 0, shape_derivs(iEle,iNode,3)];
            dsym_agrad(4,offset+1:offset+3) = [0, shape_derivs(iEle,iNode,3), shape_derivs(iEle,iNode,2)];
            dsym_agrad(5,offset+1:offset+3) = [shape_derivs(iEle,iNode,3), 0, shape_derivs(iEle,iNode,1)];
            dsym_agrad(6,offset+1:offset+3) = [shape_derivs(iEle,iNode,2), shape_derivs(iEle,iNode,1), 0];

            dskw_agrad(1,offset+1:offset+3) = [0, shape_derivs(iEle,iNode,3), -shape_derivs(iEle,iNode,2)];
            dskw_agrad(2,offset+1:offset+3) = [shape_derivs(iEle,iNode,3), 0, -shape_derivs(iEle,iNode,1)];
            dskw_agrad(3,offset+1:offset+3) = [shape_derivs(iEle,iNode,2), -shape_derivs(iEle,iNode,1), 0];

            dsym_chidd(1,offset+4:offset+12) = [all_shape_functions(iGp,iNode), 0, 0, 0, 0, 0, 0, 0, 0];
            dsym_chidd(2,offset+4:offset+12) = [0, all_shape_functions(iGp,iNode), 0, 0, 0, 0, 0, 0, 0];
            dsym_chidd(3,offset+4:offset+12) = [0, 0, all_shape_functions(iGp,iNode), 0, 0, 0, 0, 0, 0];
            dsym_chidd(4,offset+4:offset+12) = [0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, all_shape_functions(iGp,iNode), 0, 0];
            dsym_chidd(5,offset+4:offset+12) = [0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, all_shape_functions(iGp,iNode), 0];
            dsym_chidd(6,offset+4:offset+12) = [0, 0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, all_shape_functions(iGp,iNode)];

            dskw_chidd(1,offset+4:offset+12) = [0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, -all_shape_functions(iGp,iNode), 0, 0];
            dskw_chidd(2,offset+4:offset+12) = [0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, -all_shape_functions(iGp,iNode), 0];
            dskw_chidd(3,offset+4:offset+12) = [0, 0, 0, 0, 0, all_shape_functions(iGp,iNode), 0, 0, -all_shape_functions(iGp,iNode)];

            offset = offset + 12;
        end

        % inertia stress derivatives
        dsym_meso = T_e*dsym_agrad;
        dskw_meso = zeros(6,12*nodes_per_ele);
        dskw_meso(4:end,:) = T_c*dskw_agrad;
        dsym_micro = J_m*dsym_chidd;
        dskw_micro = zeros(6,12*nodes_per_ele);
        dskw_micro(4:end,:) = J_c*dskw_chidd;

        % inertia stress divergence derivatives
        dsym_inertia_divergence = zeros(12*nodes_per_ele,12*nodes_per_ele);
        dskw_inertia_divergence = zeros(12*nodes_per_ele,12*nodes_per_ele);

        offset = 0;
        for iNode = 1:nodes_per_ele
            dsym_inertia_divergence(offset+1,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*dsym_meso(1,:) + ...
                shape_derivs(iEle,iNode,2)*dsym_meso(6,:) + shape_derivs(iEle,iNode,3)*dsym_meso(5,:));
            dsym_inertia_divergence(offset+2,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*dsym_meso(6,:) + ...
                shape_derivs(iEle,iNode,2)*dsym_meso(2,:) + shape_derivs(iEle,iNode,3)*dsym_meso(4,:));
            dsym_inertia_divergence(offset+3,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,1)*dsym_meso(5,:) + ...
                shape_derivs(iEle,iNode,2)*dsym_meso(4,:) + shape_derivs(iEle,iNode,3)*dsym_meso(3,:));

            dskw_inertia_divergence(offset+1,:) = gw(iGp)*cell_volume*(shape_derivs(iEle,iNode,2)*dskw_meso(6,:) + shape_derivs(iEle,iNode,3)*dskw_meso(5,:));
            dskw_inertia_divergence(offset+2,:) = gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*dskw_meso(6,:) + shape_derivs(iEle,iNode,3)*dskw_meso(4,:));
            dskw_inertia_divergence(offset+3,:) = gw(iGp)*cell_volume*(-shape_derivs(iEle,iNode,1)*dskw_meso(5,:) - shape_derivs(iEle,iNode,2)*dskw_meso(4,:));

            offset = offset + 12;
        end
        dfull_inertia_divergence = dsym_inertia_divergence + dskw_inertia_divergence;

        % inertia stress values at nodes derivatives
        dnodal_sym_micro = zeros(12*nodes_per_ele,12*nodes_per_ele);
        dnodal_skw_micro = zeros(12*nodes_per_ele,12*nodes_per_ele);

        offset = 0;
        for iNode = 1:nodes_per_ele
            dnodal_sym_micro(offset+4:offset+6,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_micro(1:3,:);
            dnodal_sym_micro(offset+7:offset+9,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_micro(4:6,:);
            dnodal_sym_micro(offset+10:offset+12,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dsym_micro(4:6,:);

            dnodal_skw_micro(offset+4:offset+6,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dskw_micro(1:3,:);
            dnodal_skw_micro(offset+7:offset+9,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dskw_micro(4:6,:);
            dnodal_skw_micro(offset+10:offset+12,:) = -gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dskw_micro(4:6,:);

            offset = offset + 12;
        end
        dfull_nodal_inertia_stress(iEle,iGp,:,:) = dnodal_sym_micro + dnodal_skw_micro;

        % macro inertia at nodes derivatives
        dnode_inertia = zeros(12*nodes_per_ele,12*nodes_per_ele);
        dinertia_gp = zeros(3,12*nodes_per_ele);

        offset = 0;
        for iNode = 1:nodes_per_ele
            dinertia_gp(1,offset+1:offset+3) = rho*[all_shape_functions(iGp,iNode), 0, 0];
            dinertia_gp(2,offset+1:offset+3) = rho*[0, all_shape_functions(iGp,iNode), 0];
            dinertia_gp(3,offset+1:offset+3) = rho*[0, 0, all_shape_functions(iGp,iNode)];
            offset = offset + 12;
        end

        offset = 0;
        for iNode = 1:nodes_per_ele
            dnode_inertia(offset+1:offset+3,:) = gw(iGp)*cell_volume*all_shape_functions(iGp,iNode)*dinertia_gp;
            offset = offset + 12;
        end
        dtotal_inertia_residual(iEle,iGp,:,:) = dnode_inertia + dfull_inertia_divergence;
    end
end

% Assembly
node0_global_jacA = zeros(12,12);
for iEle = 1:nele
    for iGp = 1:num_gp
        local_conn = find(conn(iEle,:) == 1);
        idx = local_conn:12*local_conn;
        local_contribution = dtotal_inertia_residual(iEle, iGp, idx, idx) + dfull_nodal_inertia_stress(iEle, iGp, idx, idx);
        node0_global_jacA = node0_global_jacA + reshape(local_contribution,12,12);
    end
end