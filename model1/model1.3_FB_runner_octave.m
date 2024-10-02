% Define the number of firms
number_of_firms = 10000; % You can modify this value to change the number of firms; eventually, this will need to reflect the Firm.count parameter in model1.yaml

for i = 1:264 % 'howManyCycles' parameter in model1.yaml
    % Wait until Python has sent data
    while true
        [status, data] = system('/opt/homebrew/bin/redis-cli get python_to_octave');
        if status == 0 && !isempty(strtrim(data))
            break;
        endif
        pause(1); % Wait before checking again
    end

    % Generate an array of number_of_firms random floats between 0 and 1
    rand_floats = rand(1, number_of_firms);

    % Generate an array of number_of_firms random integers between 1 and 100
    rand_ints = randi([1, 100], 1, number_of_firms);

    % Convert arrays to strings
    float_str = sprintf('%f, ', rand_floats);
    int_str = sprintf('%d, ', rand_ints);

    % Remove the trailing comma and space
    float_str = float_str(1:end-2);
    int_str = int_str(1:end-2);    

    % Send result back to Redis
    index = strfind(data, num2str(i-1));
    index_start = strfind(data, "Start");
    if !isempty(index) || !isempty(index_start)
        % Prepare the result to send both arrays as a single string
        printf('Octave computing...')
        pause(2); % Simulate processing time - demand code will go here
        result = sprintf('Step: [%s], Floats: [%s], Integers: [%s]', num2str(i), float_str, int_str);
        printf('Octave sending arrays of number_of_firms floats and number_of_firms integers\n');
        redis_command = ['echo "', result, '" | /opt/homebrew/bin/redis-cli -x set octave_to_python'];
        system(redis_command);
    end

end
