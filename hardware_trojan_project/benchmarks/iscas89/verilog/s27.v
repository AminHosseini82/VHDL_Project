// ISCAS-89 s27 benchmark circuit
module s27(
    input clk,
    input reset,
    input [3:0] in,
    output [0:0] out
);

    reg [2:0] state, next_state;
    wire logic_out;

    // State machine
    always @(posedge clk or negedge reset) begin
        if (!reset)
            state <= 3'b000;
        else
            state <= next_state;
    end

    // Next state logic
    always @(*) begin
        case (state)
            3'b000: next_state = {in[2], in[1], in[0]};
            3'b001: next_state = {in[3], in[2], in[1]};
            3'b010: next_state = {in[0], in[3], in[2]};
            3'b011: next_state = {in[1], in[0], in[3]};
            3'b100: next_state = {in[2], in[1], in[0]};
            3'b101: next_state = {in[3], in[2], in[1]};
            3'b110: next_state = {in[0], in[3], in[2]};
            3'b111: next_state = {in[1], in[0], in[3]};
            default: next_state = 3'b000;
        endcase
    end

    // Output logic
    assign logic_out = (state[0] & in[0]) | (state[1] & in[1]) | (state[2] & in[2]);
    assign out = logic_out;

endmodule
