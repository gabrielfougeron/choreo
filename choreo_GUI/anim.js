// var Gallery_cache_behavior = {cache: "no-cache"}
var Gallery_cache_behavior = {}

var DefaultGallery_description

var UserWorkspace
var WorkspaceIsSetUp = false

var SolName 
var Pos 
var PlotInfo

var running = false;

var xMin=0., xMax=1., yMin=0., yMax=1.
var Max_PathLength = 1.

var xPixRate, yPixRate
var center_x,center_y
var CurrentMax_PartRelSize = 1.

var displayWidth, displayHeight
var trajectoriesOn

var FPS_estimation = 60
var Do_Limit_FPS = false
var FPS_limit = 120
var Elapsed_Time_During_Animation = 0
var n_valid_dt_animation = 0

var Last_UpdateFPSDisplay = 0
var UpdateFPSDisplay_freq = 5

var n_color = 0;
var colorLookup_init = [
	"#50ce4d", // Moderate Lime Green
	"#ff7006", // Vivid Orange
	"#a253c4", // Moderate Violet
	"#ef1010", // Vivid Red
	"#25b5bc", // Strong Cyan
	"#E86A96", // Soft Pink
	"#edc832", // Bright Yellow
	"#ad6530", // Dark Orange [Brown tone]
	"#00773f", // Dark cyan - lime green 
	"#d6d6d6", // Light gray
]

var defaultParticleColor = "#50ce4d"
var defaultTrailColor = "#3c9a39"
var FallbackTrailColor = "#d5d5d5"

// Particle radius
var min_base_particle_size = 1.
var max_base_particle_size = 10.
var base_particle_size = 6.

// with of particle trail
var min_base_trailWidth = 0.2
var max_base_trailWidth = 5.
var base_trailWidth = 2

// Vanish speed
var min_base_trail_vanish_length = 0.1
var max_base_trail_vanish_length = 10.
var base_trail_vanish_length = 1.
var trail_vanish_length_mul = 0.002

var DoScaleSizeWithMass = true
var DoTrailVanish = true
var LastFadeTime = 0

var SearchIsOnGoing = false

var FadeInvFrequency

var	colorLookup = colorLookup_init

var AllPosFilenames = []
var AllPlotInfoFilenames = []
var AllGalleryNames = []

var DefaultTree_Target
var Target_Tree

var Target_current_type
var Target_current_id


var TargetSlow_PlotInfo
var TargetSlow_Pos
var TargetSlow_Loaded = false

var TargetFast_PlotInfoList
var TargetFast_PosList
var	TargetFast_LoadedList

var trailColorLookup

FileSystemAccessSupported = ('showOpenFilePicker' in window)

function AjaxGet(foldername){ return $.ajax({ url: foldername})}

function GetFileBaseExt(filename) {

	var dotidx = filename.lastIndexOf('.')+1;

	var base = filename.substring(0,dotidx-1);
	var ext = filename.substring(dotidx-1, filename.length);

	return [base,ext]

}

async function ListFilesInFolder(folder){

	var FilesArray = [];

	$.ajax({
		url: folder,
		success: function(data){
			$(data).find("li > a").each(function(){
				FilesArray.push(folder+this.innerHTML);
			})
		}           
	});

	return FilesArray;

}

function windowLoadHandler() {
	canvasApp();
}

function setPlotWindow(windowObject) {

	var extend = 0.02 + (CurrentMax_PartRelSize-1.)*0.0155 ; // Magic numbers

	var xinf = windowObject.xMin - extend*(windowObject.xMax-windowObject.xMin);
	var xsup = windowObject.xMax + extend*(windowObject.xMax-windowObject.xMin);
	
	var yinf = windowObject.yMin - extend*(windowObject.yMax-windowObject.yMin);
	var ysup = windowObject.yMax + extend*(windowObject.yMax-windowObject.yMin);
	
	var hside = Math.max(xsup-xinf,ysup-yinf)/2

	center_x = (xinf+xsup)/2
	center_y = (yinf+ysup)/2

	xMin = center_x - hside
	xMax = center_x + hside

	yMin = center_y - hside
	yMax = center_y + hside

	xPixRate = displayWidth/(xMax - xMin);
	yPixRate = displayHeight/(yMin - yMax);

}

function setColorLookupList() {
		
	trailColorLookup = [];
	
	//darkening colors for trails
	var i,r,g,b,colorString,c, newColor,newColorString;
	var len = colorLookup.length;
	var darkenFactor = 0.75;
	for (i = 0; i < len; i++) {
		colorString = colorLookup[i];
		colorString = colorString.replace("#", "");
		c = parseInt(colorString,16);
		r = (c & (255 << 16)) >> 16;
		g = (c & (255 << 8)) >> 8;
		b = (c & 255); 
		
		r = Math.floor(r*darkenFactor);
		g = Math.floor(g*darkenFactor);
		b = Math.floor(b*darkenFactor);
					
		newColor = (r << 16) | (g << 8) | b;
		
		newColorString = newColor.toString(16);
		while (newColorString.length < 6) {
			newColorString = "0" + newColorString;
		}
		newColorString = "#" + newColorString;
		trailColorLookup.push(newColorString);
	}

}

function canvasApp() {
		
	var particles;

	var tInc;

	var time = 0;
	var Lasttime = time;

	var bgColor = "#F1F1F1";
	var request;

	var fadeScreenColor = "rgba(241,241,241,0.01)";
	
	var staticOrbitDrawPointsX;
	var staticOrbitDrawPointsY;
	
	var endPixX;
	var endPixY;

	var GlobalRot_angle = 0.;
	var GlobalRot = [[1.,0.],[0.,1.]];
	
	var displayCanvas = document.getElementById("displayCanvas");
	var context = displayCanvas.getContext("2d");
	context.lineCap = "round";
	displayCanvas.addEventListener("FinalizeSetOrbitFromOutsideCanvas", FinalizeSetOrbitFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("FinalizeAndPlayFromOutsideCanvas", FinalizeAndPlayFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("StopAnimationFromOutsideCanvas", StopAnimationFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("DisableAnimationFromOutsideCanvas", DisableAnimationFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("EnableAnimationFromOutsideCanvas", EnableAnimationFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("StartAnimationFromOutsideCanvas", StartAnimationFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("RemakeParticlesFromOutsideCanvas", RemakeParticlesFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("ChangeColorsFromOutsideCanvas", ChangeColorsFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("DrawAllPathsFromOutsideCanvas", DrawAllPathsFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("CompleteSetOrbitFromOutsideCanvas", CompleteSetOrbitFromOutsideCanvasHandler, true);
	
	var particleLayerCanvas = document.getElementById("particleLayerCanvas");
	var particleLayerContext = particleLayerCanvas.getContext("2d");
	particleLayerCanvas.addEventListener("click", startStopButtonHandler, true);
	
	displayWidth = displayCanvas.width;
	displayHeight = displayCanvas.height;	

	var Min_PartRelSize = 0.5;
	var Max_PartRelSize = 70.;
	
	var Last_Time_since_origin;
	var dt_outlier_ms = 1200;
	var speed_slider_value_init = .5;
	var Time_One_Period_init = 5;
	
	var startStopButton = document.getElementById("startStopButton");
	startStopButton.addEventListener("click", startStopButtonHandler, true);
	
	var trajectoryButton = document.getElementById("trajectoryButton");
	trajectoryButton.addEventListener("click", trajectoryButtonHandler, true);

	var drawTrajButton = document.getElementById("ClearButton");
	drawTrajButton.addEventListener("click", clearScreen, true);

	var speedPlusBtn = document.getElementById("speedPlusBtn");
	speedPlusBtn.addEventListener("click", SpeedPlusClick, true);
	
	var speedMinusBtn = document.getElementById("speedMinusBtn");
	speedMinusBtn.addEventListener("click", SpeedMinusClick, true);

	var speedTxt = document.getElementById("speedTxt");

	// http://paulirish.com/2011/requestanimationframe-for-smart-animating/
	// http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating
	
	// requestAnimationFrame polyfill by Erik Möller
	// fixes from Paul Irish and Tino Zijdel
	
	(function() {
		var the_lastTime = 0;
		var vendors = ['ms', 'moz', 'webkit', 'o'];
		for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
			window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
			window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame']
									|| window[vendors[x]+'CancelRequestAnimationFrame'];
		}
	
		if (!window.requestAnimationFrame)
			window.requestAnimationFrame = function(callback, element) {
				var currTime = new Date().getTime();
				var timeToCall = Math.max(0, 16 - (currTime - the_lastTime));
				var id = window.setTimeout(function() { callback(currTime + timeToCall); },
				timeToCall);
				the_lastTime = currTime + timeToCall;
				return id;
			};
	
		if (!window.cancelAnimationFrame)
			window.cancelAnimationFrame = function(id) {
				clearTimeout(id);
			};
	}());
	
	init();
	
	async function init() {
		
		setColorLookupList()

		trajectoriesOn = true;
		
		//jquery ui elements
		//speed slider
		$("#speedSlider").slider({
		  value: speed_slider_value_init,
		  orientation: "horizontal",
		  range: "min",
		  max: 1,
		  min: 0.2,
		  step: 0.005,
		  input:speedSliderHandler,
		  change:speedSliderHandler,
		  animate: false,
		});		

		//Rotation slider
		$("#RotSlider").roundSlider({
				width: 15,
				radius: 50,
				value: 0,
				keyboardAction: false,
				lineCap: "square",
				startAngle: 90,
				max: 360,
				mouseScrollAction: true,
				step: 5,
				// editableTooltip: false,
				// change:onRotationChange,
				valueChange:onRotationValueChange,
				disabled:false,
			});
		
		// Load the static gallery
		await LoadDefaultGallery()
		startAnimation()

	}

	function Estimate_FPS(Time_since_origin){

		// console.log(FPS_estimation)

		if (Last_Time_since_origin && Time_since_origin)
		{
			var dt_ms = Time_since_origin - Last_Time_since_origin;

			if ( dt_ms < dt_outlier_ms)
			{
				Elapsed_Time_During_Animation += dt_ms/1000;
				n_valid_dt_animation +=1;

				FPS_estimation = n_valid_dt_animation/Elapsed_Time_During_Animation;

			}
		
		}
	
		Last_Time_since_origin = Time_since_origin;	

		setPeriodTime()

	}

	function anim_particles() {
		
		UpdateFPSDisplay();
		onTimer();
	}

	function anim_particles_loop(Time_since_origin){

		request = requestAnimationFrame(anim_particles_loop);

		if (Do_Limit_FPS) {

			if (anim_schedule_time < Time_since_origin) {
				anim_schedule_time += 1000/FPS_limit ;
				Estimate_FPS(Time_since_origin);
				anim_particles();
			}

		} else {
			Estimate_FPS(Time_since_origin);
			anim_particles();
		}

	}
	
	function startAnimation() {
		running = true;
		startStopButton.textContent = "Stop";
		input_Limit_FPS_Handler();
		anim_particles_loop();
	}

	function stopAnimation() {
		running = false;
		cancelAnimationFrame(request);
		startStopButton.textContent = "Start";
	}
	
	function startStopButtonHandler(e) {

		if (!document.getElementById("startStopButton").disabled) {
			if (running) {
				stopAnimation();
			}
			else {
				startAnimation();
			}
		}
	}

	function trajectoryButtonHandler(e) {
		if (trajectoriesOn) {
			trajectoriesOn = false;
			trajectoryButton.textContent ="Draw trails";
			clearScreen();
		}
		else {
			setStartPositions();
			trajectoriesOn = true;
			trajectoryButton.textContent = "Hide trails";
		}
	}

	function StopAnimationFromOutsideCanvasHandler(e) {
		stopAnimation();
	}

	function DisableAnimationFromOutsideCanvasHandler(e) {
		DisableAnimation();
		clearScreen();
		clearParticleLayer();
	}

	function DisableAnimation() {

		document.getElementById("startStopButton").disabled = "disabled";

	}

	function EnableAnimationFromOutsideCanvasHandler(e) {
		
		EnableAnimation();

	}

	function EnableAnimation() {

		document.getElementById("startStopButton").disabled = "";

	}

	function StartAnimationFromOutsideCanvasHandler(e) {
		startAnimation();
	}

	function FinalizeSetOrbitFromOutsideCanvasHandler(e) {

		clearScreen();
		clearParticleLayer();
		FinalizeSetOrbit(DoDrawParticles=false,DoXMinMax=true,setTinc=false) ;

		if (document.getElementById('checkbox_DisplayBodiesDuringSearch').checked) {

			startAnimation();
	
		}

	}

	function CompleteSetOrbitFromOutsideCanvasHandler(e) {
		CompleteSetOrbit()
	}

	function FinalizeAndPlayFromOutsideCanvasHandler(e) {

		if (e.DoClearScreen) {
			clearScreen();
			
			if (trajectoriesOn && document.getElementById('checkbox_DisplayLoopOnGalleryLoad').checked){

				if (DoTrailVanish) {
					request = requestAnimationFrame(anim_path_grey)
				} else {
					request = requestAnimationFrame(anim_path)
				}
				
			}
		}
		FinalizeSetOrbit(DoDrawParticles=true,DoXMinMax = e.DoXMinMax,setTinc=e.setTinc ) ;
		startAnimation();
	}

	function RemakeParticlesFromOutsideCanvasHandler(e) {
		makeParticles();
		clearScreen();
		clearParticleLayer();
		drawParticles();
	}

	function ChangeColorsFromOutsideCanvasHandler(e) {
		
		setColorLookupList()
		makeParticles();
		clearScreen();
		clearParticleLayer();
		drawParticles();
	}

	function onRotationValueChange(e){

		if ((e.action == 'code') || (e.action == 'change') || (e.action == 'drag')) {

			GlobalRot_angle = e.value * 2* Math.PI / 360.;
			GlobalRot = [
				[ Math.cos(GlobalRot_angle), Math.sin(GlobalRot_angle)],
				[-Math.sin(GlobalRot_angle), Math.cos(GlobalRot_angle)]
			]

			var delta_angle = (e.value - e.preValue)* 2* Math.PI / 360.;
			
			RotateCanvas(displayCanvas,context,delta_angle);
			
			setParticlePositions(time);
			setParticlePositions(time); // dirty hack
			clearParticleLayer();
			drawParticles();

		}
	}

	function RotateCanvas(canvas,ctx,angle){

		var tempCanvas = document.createElement("canvas");
		var tempCtx = tempCanvas.getContext("2d");

		tempCanvas.width = canvas.width;
		tempCanvas.height = canvas.height;
		tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height);

		ctx.save(); //saves the state of canvas
		ctx.fillStyle = bgColor;
		ctx.fillRect(0,0,canvas.width,canvas.height);

		ctx.translate(
			canvas.width * 0.5,
			canvas.height * 0.5
		);
		ctx.rotate(angle);
		ctx.translate(
			-canvas.width * 0.5,
			-canvas.height * 0.5
		);
		ctx.drawImage(tempCanvas,  0, 0, canvas.width, canvas.height);
		ctx.restore();

	};

	function onTimer() {

		if (trajectoriesOn) {
			
			//fade
			context.fillStyle = fadeScreenColor

			
			var nfade = Math.floor(LastFadeTime/FadeInvFrequency)
			if (!DoTrailVanish) {nfade = 0}

			for (var ifade=0; ifade<nfade; ifade++){
				context.fillRect(0,0,displayWidth,displayHeight)
			}

			LastFadeTime = LastFadeTime + tInc - nfade*FadeInvFrequency

		}
		
		//clear particle layer
		clearParticleLayer()
		
		Lasttime = time
		time = (time + tInc) % 1
		
		//update particles
		setParticlePositions(time)
		
		//draw particles
		drawParticles()

		if (trajectoriesOn) {
			if (Pos !== undefined) {
				drawPathPortion(Lasttime,tInc) 
			}
		}
	}
	
	function clearScreen() {
		context.fillStyle = bgColor;
		context.fillRect(0,0,displayWidth,displayHeight);
	}
	
	function clearParticleLayer() {
		particleLayerContext.clearRect(0,0,displayWidth+1,displayHeight+1);
	}
	
	function makeParticles() {

		particles = new Array(PlotInfo['nbody'])

		CurrentMax_PartRelSize = 0
		
		var color_method_input = document.getElementById("color_method_input")

        for ( var il = 0 ; il < PlotInfo['nloop'] ; il++){

			var nlb =  PlotInfo['loopnb'][il]
			
			for (var ilb = 0 ; ilb < nlb ; ilb++){

				var ib = PlotInfo['Targets'][il][ilb];

				var color;
				var trailColor;

				var color_id;

				if (color_method_input.value == "body") {
					color_id = ib;
				} else if (color_method_input.value == "loop") {
					color_id = il;
				} else if (color_method_input.value == "loop_id") {
					color_id = ilb;
				} else {
					color_id = 0;
				}
				
				color = colorLookup[color_id % colorLookup.length];
				trailColor = trailColorLookup[color_id % colorLookup.length];

				if (DoScaleSizeWithMass) {
					var PartRelSize = Math.sqrt(PlotInfo["mass"][ib]);
				} else {
					var PartRelSize = 1.;
				}

				PartRelSize = Math.min(PartRelSize,Max_PartRelSize);
				PartRelSize = Math.max(PartRelSize,Min_PartRelSize);
				// Min/max ?
				
				CurrentMax_PartRelSize = Math.max(CurrentMax_PartRelSize,PartRelSize);
						
				particles[ib] = {
						x: 0,
						y: 0,
						lastX: 0,
						lastY: 0,
						color: color,
						trailColor: trailColor,
						PartRelSize: PartRelSize,
				}

			}
		}

		setParticlePositions(time);
		resetLastPositions();
	}
	
	function resetLastPositions() {
		//set initial last positions
		for (i = 0; i<PlotInfo["nbody"]; i++) {
			particles[i].lastX = particles[i].x;
			particles[i].lastY = particles[i].y;
		}
	}
	
	function setStartPositions() {
		var pixX;
		var pixY;
		var j;
		endPixX = [];
		endPixY = [];
		staticOrbitDrawPointsX = [];
		staticOrbitDrawPointsY = [];
		for (i = 0; i<PlotInfo["nbody"]; i++) {
			j = (i + 1) % PlotInfo["nbody"];
			pixX = xPixRate*(particles[j].x - xMin);
			pixY = yPixRate*(particles[j].y - yMax);
			endPixX.push(pixX);
			endPixY.push(pixY);
			staticOrbitDrawPointsX.push(xPixRate*(particles[i].x - xMin));
			staticOrbitDrawPointsY.push(yPixRate*(particles[i].y - yMax));
		}
	}

	function anim_path(Time_since_origin){

		clearScreen();
		DrawAllPaths();

	}

	function DrawAllPathsFromOutsideCanvasHandler() {

		if (running) {
			cancelAnimationFrame(request)
		}

		request = requestAnimationFrame(anim_path)

		if (running) {
			request = requestAnimationFrame(anim_particles_loop)
		}

	}

	function DrawAllPaths() {

		var n_pos = Pos.shape[2];

		var xl,yl;
		var x,y;
		var Pixx,Pixy;

		var il,ib,ilb,nlb;
		var p;

        for ( il = 0 ; il < PlotInfo['nloop'] ; il++){

			nlb =  PlotInfo['loopnb'][il];
			
			for (ilb = 0 ; ilb < nlb ; ilb++){

				if (PlotInfo['RequiresLoopDispUn'][il][ilb]) {

					ib = PlotInfo['Targets'][il][ilb];
					p = particles[ib];
					context.lineWidth = p.PartRelSize * base_trailWidth ;
					context.strokeStyle = p.trailColor;

					// Super ugly
					xl = Pos.data[  2*il    * n_pos] ;
					yl = Pos.data[ (2*il+1) * n_pos] ;

					x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
					y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;

					Pixx = xPixRate*(x - xMin) ;
					Pixy = yPixRate*(y - yMax) ;

					context.beginPath();
					context.moveTo(Pixx, Pixy);

					for (i_pos = 1 ; i_pos < n_pos ; i_pos++){

						// Super ugly
						xl = Pos.data[ i_pos +  2*il    * n_pos] ;
						yl = Pos.data[ i_pos + (2*il+1) * n_pos] ;

						x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
						y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;
		
						Pixx = xPixRate*(x - xMin) ;
						Pixy = yPixRate*(y - yMax) ;

						context.lineTo(Pixx, Pixy);

					}

					// Super ugly
					xl = Pos.data[  2*il    * n_pos] ;
					yl = Pos.data[ (2*il+1) * n_pos] ;

					x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
					y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;

					Pixx = xPixRate*(x - xMin) ;
					Pixy = yPixRate*(y - yMax) ;

					context.lineTo(Pixx, Pixy);
					context.stroke();

				}

			}

		}

	}
	
	function anim_path_grey(Time_since_origin){

		clearScreen();
		DrawAllPaths_Grey();

	}

	function DrawAllPaths_Grey() {

		var n_pos = Pos.shape[2];

		var xl,yl;
		var x,y;
		var Pixx,Pixy;

		var il,ib,ilb,nlb;
		var p;

        for ( il = 0 ; il < PlotInfo['nloop'] ; il++){

			nlb =  PlotInfo['loopnb'][il];
			
			for (ilb = 0 ; ilb < nlb ; ilb++){

				if (PlotInfo['RequiresLoopDispUn'][il][ilb]) {

					ib = PlotInfo['Targets'][il][ilb];
					p = particles[ib];
					context.lineWidth = p.PartRelSize * base_trailWidth ;
					context.strokeStyle = FallbackTrailColor;

					// Super ugly
					xl = Pos.data[  2*il    * n_pos] ;
					yl = Pos.data[ (2*il+1) * n_pos] ;

					x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
					y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;

					Pixx = xPixRate*(x - xMin) ;
					Pixy = yPixRate*(y - yMax) ;

					context.beginPath();
					context.moveTo(Pixx, Pixy);

					for (i_pos = 1 ; i_pos < n_pos ; i_pos++){

						// Super ugly
						xl = Pos.data[ i_pos +  2*il    * n_pos] ;
						yl = Pos.data[ i_pos + (2*il+1) * n_pos] ;

						x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
						y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;
		
						Pixx = xPixRate*(x - xMin) ;
						Pixy = yPixRate*(y - yMax) ;

						context.lineTo(Pixx, Pixy);

					}

					// Super ugly
					xl = Pos.data[  2*il    * n_pos] ;
					yl = Pos.data[ (2*il+1) * n_pos] ;

					x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
					y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;

					Pixx = xPixRate*(x - xMin) ;
					Pixy = yPixRate*(y - yMax) ;

					context.lineTo(Pixx, Pixy);
					context.stroke();

				}

			}

		}

	}

	function drawParticles() {
		var i;
		var len;
		var pixX;
		var pixY;
		var p;
		
		len = particles.length;
		for (i = 0; i < len; i++) {
			p = particles[i];
			pixX = xPixRate*(p.x - xMin);
			pixY = yPixRate*(p.y - yMax);
			lastPixX = xPixRate*(p.lastX - xMin);
			lastPixY = yPixRate*(p.lastY - yMax);
						
			var size = p.PartRelSize * base_particle_size

			//particle
			particleLayerContext.strokeStyle = "rgba(0,0,0,0.5)"
			particleLayerContext.lineWidth = 2;
			particleLayerContext.fillStyle = p.color;
			particleLayerContext.beginPath();
			particleLayerContext.arc(pixX,pixY,size,0,Math.PI*2,false);
			particleLayerContext.closePath();
			particleLayerContext.fill();
			particleLayerContext.stroke();

		}

	}

	function drawPathPortion(lasttime,tInc) {

		var n_pos = Pos.shape[2]

		var xi=0,yi=0
		var xmid,ymid
		var xrot,yrot
		var xrot_glob,yrot_glob
		var p

		var il,ib,ilb,nlb
		var tb_beg, tb_end

		var i,di
		var ip,ip_beg,ip_end

        for ( il = 0 ; il < PlotInfo['nloop'] ; il++){

			nlb =  PlotInfo['loopnb'][il]
			
			for (ilb = 0 ; ilb < nlb ; ilb++){

				ib = PlotInfo['Targets'][il][ilb]
				p = particles[ib]
				context.strokeStyle = p.trailColor
				context.lineWidth = p.PartRelSize * base_trailWidth 
				lastPixX = xPixRate*(p.lastX - xMin)
				lastPixY = yPixRate*(p.lastY - yMax)
				context.beginPath()
				context.moveTo(lastPixX,lastPixY)

				tb_beg = ( PlotInfo['TimeRevsUn'][il][ilb] * (lasttime - PlotInfo['TimeShiftNumUn'][il][ilb] / PlotInfo['TimeShiftDenUn'][il][ilb]) +1)
				ip_beg = (Math.floor(tb_beg*n_pos)+1)

				tb_end = ( PlotInfo['TimeRevsUn'][il][ilb] * ((lasttime+tInc) - PlotInfo['TimeShiftNumUn'][il][ilb] / PlotInfo['TimeShiftDenUn'][il][ilb]) +1)
				ip_end = (Math.floor(tb_end*n_pos)+1)

				di = PlotInfo['TimeRevsUn'][il][ilb]

				for (i = ip_beg ; i != ip_end ; i += di){ 

					ip = (((i%n_pos) + n_pos) % n_pos)

					// Super ugly
					xi = Pos.data[ ip +  2*il    * n_pos] 
					yi = Pos.data[ ip + (2*il+1) * n_pos] 

					xrot = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xi + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yi - center_x
					yrot = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xi + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yi - center_y

					xrot_glob = center_x + GlobalRot[0][0] * xrot + GlobalRot[0][1] * yrot
					yrot_glob = center_y + GlobalRot[1][0] * xrot + GlobalRot[1][1] * yrot

					PixX = xPixRate*(xrot_glob - xMin)
					PixY = yPixRate*(yrot_glob - yMax)

					context.lineTo(PixX, PixY)
					
					lastPixX = PixX
					lastPixY = PixY

				}

				PixX = xPixRate*(p.x - xMin);
				PixY = yPixRate*(p.y - yMax);

				context.lineTo(PixX, PixY)
				context.stroke()


			}
		
		}

	}
	
	function setParticlePositions(t) {

		var n_pos = Pos.shape[2];

		var xlm=0,xlp=0,ylm=0,ylp=0;
		var xmid,ymid;
		var xrot,yrot;
		var xrot_glob,yrot_glob;

		var il,ib,ilb,nlb;
		var tb;

		var im,ip,tbn,trem;

        for ( il = 0 ; il < PlotInfo['nloop'] ; il++){

			nlb =  PlotInfo['loopnb'][il];
			
			for (ilb = 0 ; ilb < nlb ; ilb++){

				ib = PlotInfo['Targets'][il][ilb];

				tb = ( PlotInfo['TimeRevsUn'][il][ilb] * (t - PlotInfo['TimeShiftNumUn'][il][ilb] / PlotInfo['TimeShiftDenUn'][il][ilb]) +1) % 1;

				tbn = tb*n_pos;
				
				im = Math.floor(tbn);
				ip = (im+1) % n_pos;
				
				trem = tbn - im;

				// Super ugly
				xlm = Pos.data[ im +  2*il    * n_pos] ;
				ylm = Pos.data[ im + (2*il+1) * n_pos] ;
				xlp = Pos.data[ ip +  2*il    * n_pos] ;
				ylp = Pos.data[ ip + (2*il+1) * n_pos] ;
			
				xmid = (1-trem) * xlm + trem * xlp;
				ymid = (1-trem) * ylm + trem * ylp;

				xrot = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xmid + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * ymid ;
				yrot = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xmid + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * ymid ;

				xrot = xrot - center_x;
				yrot = yrot - center_y;

				xrot_glob = center_x + GlobalRot[0][0] * xrot + GlobalRot[0][1] * yrot
				yrot_glob = center_y + GlobalRot[1][0] * xrot + GlobalRot[1][1] * yrot
				
				particles[ib].lastX = particles[ib].x;
				particles[ib].lastY = particles[ib].y;
				particles[ib].x = xrot_glob;
				particles[ib].y = yrot_glob;
				
			}
		
		}

	}

	function speedSliderHandler() {
		setPeriodTime();	
	}
	
	function SpeedMinusClick() {
		var speedSlider = $("#speedSlider");
		var cur_val = speedSlider.slider("value");
		var min_val = speedSlider.slider("option", "min");
		var max_val = speedSlider.slider("option", "max");
		var step_val = speedSlider.slider("option", "step");

		target_val = cur_val - step_val;

		if (target_val > min_val ) {
			speedSlider.slider("value",target_val);
		}

		setPeriodTime();

	}

	function SpeedPlusClick() {
		var speedSlider = $("#speedSlider");
		var cur_val = speedSlider.slider("value");
		var min_val = speedSlider.slider("option", "min");
		var max_val = speedSlider.slider("option", "max");
		var step_val = speedSlider.slider("option", "step");

		target_val = cur_val + step_val;

		if (target_val < max_val ) {
			speedSlider.slider("value",target_val);
		}

		setPeriodTime();

	}

	function setPeriodTime() {

		var slider_value = $("#speedSlider").slider("value")
		var slider_value_rel = slider_value/speed_slider_value_init
		var sliderpow = 3
		var alpha = Math.pow(slider_value_rel,sliderpow)

		var Time_One_Period = Time_One_Period_init / alpha

		var dx = xMax - xMin
		var dy = yMax - yMin
		var distance_ref = Math.sqrt(dx*dx + dy*dy)
		var distance_rel = Max_PathLength / distance_ref

		tInc = 1/(Time_One_Period*FPS_estimation * distance_rel) 

		var speedTxt_val = Math.round(100*Math.pow(slider_value,sliderpow));
		speedTxt.innerHTML = "Speed: "+speedTxt_val.toString();

	}

	function FinalizeSetOrbit(DoDrawParticles=true,DoXMinMax=true,setTinc=true) {

		if(DoXMinMax) {
			plotWindow = {
				xMin : PlotInfo["xinf"],
				xMax : PlotInfo["xsup"],
				yMin : PlotInfo["yinf"],
				yMax : PlotInfo["ysup"],
			}

			setPlotWindow(plotWindow);
		}

		SlideTrailTime();
		makeParticles();

		setParticlePositions(time);
		resetLastPositions();
		setStartPositions();
		
		//if stopped, draw particles in correct place
		if (!running) {
			clearParticleLayer();
			if (DoDrawParticles) {
				drawParticles();	
			}
		}
		if (setTinc) {
			setPeriodTime();
		}
		
	}

	function CompleteSetOrbit() {
		
		clearScreen();
		FinalizeSetOrbit();

		if (trajectoriesOn && document.getElementById('checkbox_DisplayLoopOnGalleryLoad').checked){

			if (DoTrailVanish) {
				request = requestAnimationFrame(anim_path_grey)
			} else {
				request = requestAnimationFrame(anim_path)
			}
			
		}

	}

	

// 
// 	document.onkeyup = function (event) {
// 
// 		switch (event.code) {
// 			case 'Space':
// 				startStopButtonHandler();
// 				break;
// 			case 'Enter':
// 			case 'NumpadEnter':
// 				ChoreoExecuteClick();
// 				break;
// 			case 'ArrowRight':
// 				SpeedPlusClick();
// 				break;
// 			case 'ArrowDown':
// 				SpeedMinusClick();
// 				break;
// 
// 		  }
// 
// 	}

}
