function [Best_Score, Best_Pos, CG_curve] = NRBO(N, MaxIt, lb, ub, dim, fobj)

    % Input arguments:
    % N     - Number of particles in the population 种群粒子数
    % MaxIt - Maximum number of iterations   最大迭代次数
    % lb    - Lower bound of the search space  参数低边界--搜寻空间？
    % ub    - Upper bound of the search space   参数高边界
    % dim   - Dimensionality of the search space  参数个数
    % fobj  - Objective function to minimize/maximize

    % Deciding Factor for Trap Avoidance Operator
    DF = 0.6;

    % Initialize the bounds for each dimension
%     lb = ones(1, dim) * lb;           
%     ub = ones(1, dim) * ub;
    ub=ub.*ones(1,dim);
    lb=lb.*ones(1,dim);
    % Initialization of the population
    Position = initialization(N, dim, ub, lb);  %猴子过山，10个，初始的10个，上下界范围内随机生成 10个，每个是3个参数
    Fitness = zeros(N, 1); % Vector to store individual costs  初始适应度为0

    % Calculate the initial fitness for each particle--初始10个的适应度
    for i = 1:N
        Fitness(i) = fobj(Position(i,:));   %传过去初始种群，查看适应度   
    end

    % Determine the best and worst fitness in the initial population
    [~, Ind] = sort(Fitness);     
    Best_Score = Fitness(Ind(1));
    Best_Pos = Position(Ind(1),:);
    Worst_Cost = Fitness(Ind(end));
    Worst_Pos = Position(Ind(end),:);

    % Initialize convergence curve
    CG_curve = zeros(1, MaxIt);

    % Main optimization loop
    for it = 1:MaxIt  %迭代
        % Dynamic parameter delta, decreases over iterations
        delta = (1 - ((2 * it) / MaxIt)) .^ 5;  %公式15

        % Loop over all particles in the population
        for i = 1:N        %10个里面优选          
            % Randomly select two different indices for differential evolution
            P1 = randperm(N, 2);                                       
            a1 = P1(1); a2 = P1(2);

            % Calculate the step size rho
            rho = rand * (Best_Pos - Position(i,:)) + rand * (Position(a1,:) - Position(a2,:));

            % Apply Newton-Raphson Search Rule  %应用搜寻规则
            Flag = 1;                   
            NRSR = SearchRule(Best_Pos, Worst_Pos, Position(i,:), rho, Flag);      
            X1 = Position(i,:) - NRSR + rho;                                  
            X2 = Best_Pos - NRSR + rho;                                            

            % Update position of particle
            Xupdate = zeros(1, dim);
            for j = 1:dim                                                                       
                X3 = Position(i,j) - delta * (X2(j) - X1(j));           
                a1 = rand; a2 = rand;
                Xupdate(j) = a1 * (a1 * X1(j) + (1 - a2) * X2(j)) + (1 - a2) * X3;             
            end

            % Trap Avoidance Operator to prevent local optima %防止局部最优化
            if rand < DF
                theta1 = -1 + 2 * rand(); theta2 = -0.5 + rand();      
                beta = rand < 0.5;
                u1 = beta * 3 * rand + (1 - beta); u2 = beta * rand + (1 - beta);          
                if u1 < 0.5
                    X_TAO = Xupdate +  theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));
                else
                    X_TAO = Best_Pos + theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));  
                end
                Xnew = X_TAO;
            else
                Xnew = Xupdate;
            end

            % Enforce boundary conditions
            Xnew = min(max(Xnew, lb), ub);

            % Evaluate new solution
            Xnew_Cost = fobj(Xnew);

            % Update the best and worst positions
            if Xnew_Cost < Fitness(i)
                Position(i,:) = Xnew;
                Fitness(i) = Xnew_Cost;

                % Update the global best solution
                if Fitness(i) < Best_Score
                    Best_Pos = Position(i,:);
                    Best_Score = Fitness(i);
                end
            end

            % Update the global worst solution
            if Fitness(i) > Worst_Cost
                Worst_Pos = Position(i,:);
                Worst_Cost = Fitness(i);
            end
        end

        % Update convergence curve
        CG_curve(it) = Best_Score;

        % Display iteration information
        disp(['Iteration ' num2str(it) ': Best Fitness = ' num2str(CG_curve(it))]);
        % IFBest Fitness不变化了就跳出去吧-MaxIt可变的
    end
end