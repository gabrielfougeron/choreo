
function trace(message) {
	try {
		console.log(message);
	}
	catch (exception) {
		return;
	}
}

function windowLoadHandler() {
	canvasApp();
}

function canvasApp() {
		
	var particles;
	var numParticles;
	var orbitName;
	var jsonData;
	var numOrbits;
	var xSinFreq;
	var xCosFreq;
	var ySinFreq;
	var yCosFreq;
	var xSinCoeff;
	var xCosCoeff;
	var ySinCoeff;
	var yCosCoeff;
	var tInc;
	var tIncMin, tIncMax;
	var xMin, xMax, yMin, yMax;
	var xPixRate, yPixRate;
	var time;
	var particleRad;
	var bgColor;
	var request;
	var running;
	var fadeAlpha;
	var trailWidth;
	var trajectoriesOn;
	var colorLookup;
	var defaultParticleColor;
	var timeFactor;
	var staticOrbitColor;
	var staticOrbitWidth;
	var trailColorLookup;
	
	var orbitDrawStartTime;
	var orbitDrawTime;
	var drawingStaticOrbit;
	
	var staticOrbitDrawPointsX;
	var staticOrbitDrawPointsY;
	
	var staticOrbitMinDrawDistance;
	
	var endPixX;
	var endPixY;
	
	var displayCanvas = document.getElementById("displayCanvas");
	var context = displayCanvas.getContext("2d");
	
	var particleLayerCanvas = document.getElementById("particleLayerCanvas");
	var particleLayerContext = particleLayerCanvas.getContext("2d");
	
	var orbitLayerCanvas = document.getElementById("orbitLayerCanvas");
	var orbitLayerContext = orbitLayerCanvas.getContext("2d");
	
	var displayWidth = displayCanvas.width;
	var displayHeight = displayCanvas.height;
	
	var startStopButton = document.getElementById("startStopButton");
	startStopButton.addEventListener("click", startStopButtonHandler, true);
	
	var trajectoryButton = document.getElementById("trajectoryButton");
	trajectoryButton.addEventListener("click", trajectoryButtonHandler, true);
	
	var btnNextOrbit = document.getElementById("btnNextOrbit");
	btnNextOrbit.addEventListener("click", nextOrbit, true);
	
	var btnPrevOrbit = document.getElementById("btnPrevOrbit");
	btnPrevOrbit.addEventListener("click", prevOrbit, true);
	
	//requestAnimationFrame shim for multiple browser compatibility by Eric Möller,
	//http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating
	//For an alternate version, also see http://www.paulirish.com/2011/requestanimationframe-for-smart-animating/ 
	(function() {
		var lastTime = 0;
		var vendors = ['ms', 'moz', 'webkit', 'o'];
		for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
			window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
			window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame'] 
									   || window[vendors[x]+'CancelRequestAnimationFrame'];
		}
	 
		if (!window.requestAnimationFrame)
			window.requestAnimationFrame = function(callback, element) {
				var currTime = new Date().getTime();
				var timeToCall = Math.max(0, 16 - (currTime - lastTime));
				var id = window.setTimeout(function() { callback(currTime + timeToCall); }, 
				  timeToCall);
				lastTime = currTime + timeToCall;
				return id;
			};
	 
		if (!window.cancelAnimationFrame)
			window.cancelAnimationFrame = function(id) {
				clearTimeout(id);
			};
	}());
	
	init();
	
	function init() {
		particleRad = 5.5;
		trailWidth = 2;
		// bgColor = "#000000";
		bgColor = "#F1F1F1";
		fadeScreenColor = "rgba(255,255,255,0.02)";
		// fadeScreenColor = "rgba(0,0,0,0)";
		// staticOrbitColor = "rgba(130,180,270,0.3)";
		staticOrbitColor = "rgba(130,180,270,0.3)";
		// staticOrbitColor = "rgba(255,0,255,0.8)"; //TESTING
		staticOrbitWidth = trailWidth;
		defaultParticleColor = "#ee6600";
		defaultTrailColor = "#dd5500";
		
		staticOrbitMinDrawDistance = 2;
		
		setColorLookupList();
		
		//The line below is commented out because I decided to keep it simple and load the JSON within a separate JavaScript file.
		//getJSON("test.json", setData);
		
		setData(testData);
		
		trajectoriesOn = true;
		drawingStaticOrbit = true;
		
		//jquery ui elements
		//speed slider
		$("#speedSlider").slider({
		  value: 0.33,
		  orientation: "horizontal",
		  range: "min",
		  max: 1,
		  step: 0.005,
		  slide: speedSliderHandler,
		  change: speedSliderHandler,
		  animate: true
		});
		
		tIncMin = 0.001;
		tIncMax = 0.07;

		//set first orbit (includes makeParticles)
		setOrbit(0);
		
		orbitDrawStartTime = orbitDrawTime = time = 0;
		
		startAnimation();
	}
	
	function setColorLookupList() {
		colorLookup = ["#ff7006","#50ce4d","#a253c4","#ef1010","#25b5bc","#E86A96","#edc832","#ad6530","#00773f","#d6d6d6"];
		
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
	
	function incrementOrbit(inc) {
		//find out what is checked
		currentIndex = $('input[name=orbitGroup]:checked').index('input[name=orbitGroup]');
		currentIndex = (currentIndex + inc + numOrbits) % numOrbits;
		
		//make new selection
		$('input[name=orbitGroup]')[currentIndex].checked = true;
		$("#orbitRadio").buttonset('refresh');
		
		//scroll
		//how much is currently scrolled:
		var currentScroll = $('#radioContainer').scrollTop();
		//the current position of the selected radio button:
		var rowPos = $('input[name=orbitGroup]:checked').position();
		//rollTop sets how many pixels of area to be above viewable area:
		var scrollAmount = currentScroll + rowPos.top;
		//stop any currently running animations:
		$('#radioContainer').stop();
		//animate scroll:
		$('#radioContainer').animate({scrollTop:scrollAmount + "px"});
		
		
		//set orbit
		setOrbit(currentIndex);
	}
	
	function nextOrbit(evt) {
		incrementOrbit(1);
	}
	function prevOrbit(evt) {
		incrementOrbit(-1);
	}
	
	function setPlotWindow(windowObject) {
		xMin = windowObject.xMin;
		xMax = windowObject.xMax;
		yMin = windowObject.yMin;
		yMax = windowObject.yMax;
		xPixRate = displayWidth/(xMax - xMin);
		yPixRate = displayHeight/(yMin - yMax);
	}
	
	function populateOrbitRadioButtons(dataObject) {
		var i;
		numOrbits = dataObject.orbits.length;
		var orbitRadio = document.getElementById("orbitRadio");
		for (i = 0; i < numOrbits; i++) {
			
			//radio button
			var input = document.createElement('input');
			input.type = "radio";
    		input.value = i;
			input.id = "radio"+i;
			if (i == 0) {
				input.checked = "checked";
			}
			input.name = "orbitGroup";

			//label for the button
			var label = document.createElement('label');
			label.id = "label"+i;
			label.setAttribute("for",input.id);
			input.setAttribute("mylabel",label.id)
			label.className = "radioLabel w3-button w3-hover-pale-red w3-light-grey ";
			label.innerHTML = dataObject.orbits[i].name;
			
			//add to DOM
			orbitRadio.appendChild(input);
    		orbitRadio.appendChild(label);
			orbitRadio.appendChild(document.createElement('br'));
		}
		
		//jQuery: convert radio button group to buttonset.
		$("#orbitRadio").buttonset();

		$("input[name='orbitGroup']").on('change',
        	function() {

				$("label").filter(".w3-red").each(function(i, obj) {
					obj.classList.remove('w3-red');
					obj.classList.add('w3-light-grey');
				});

				mylabel = this.getAttribute("mylabel")
				thelabel=$("label[id="+mylabel+"]")["0"];			
				thelabel.classList.add('w3-red');
				thelabel.classList.remove('w3-light-grey');

				setOrbit(this.value);
        	}
    	);	

		$('label:first', "#orbitRadio").removeClass('w3-light-grey').addClass('w3-red');
		//changes button rounded corner style to fit vertical alignment style
		$('label:first', "#orbitRadio").removeClass('ui-corner-left').addClass('ui-corner-top').addClass('w3-red');
		$('label:last', "#orbitRadio").removeClass('ui-corner-right').addClass('ui-corner-bottom');

	}
	
	function startAnimation() {
		running = true;
		(function animloop(){
		  request = requestAnimationFrame(animloop);
		  onTimer();
		})();
	}
	function stopAnimation() {
		running = false;
		cancelAnimationFrame(request);
	}
	
	function startStopButtonHandler(e) {
		if (running) {
			stopAnimation();
			running = false;
			startStopButton.textContent = "Start";
		}
		else {
			startAnimation();
			running = true;
			startStopButton.textContent = "Stop";
		}
	}
	
	function trajectoryButtonHandler(e) {
		if (trajectoriesOn) {
			trajectoriesOn = false;
			drawingStaticOrbit = false;
			trajectoryButton.textContent ="Show trajectories";
			clearScreen();
		}
		else {
			orbitDrawStartTime = orbitDrawTime = time;
			drawingStaticOrbit = true;
			setStartPositions();
			trajectoriesOn = true;
			trajectoryButton.textContent = "Hide trajectories";
		}
	}

	
	function onTimer() {
		if (trajectoriesOn) {
			//fade
			context.fillStyle = fadeScreenColor;
			context.fillRect(0,0,displayWidth,displayHeight);
		}
		
		//clear particle layer
		clearParticleLayer();
		
		if (drawingStaticOrbit) {
			orbitDrawTime += tInc;
			if (orbitDrawTime > orbitDrawStartTime + 2*Math.PI/numParticles) {
				//stop drawing orbit
				drawingStaticOrbit = false;
				//draw last segments
				drawLastSegments();
			}
		}		
		
		time = (time + tInc) % (2*Math.PI);
		
		//update particles
		setParticlePositions(time);
		
		//draw particles
		drawParticles();
	}
	
	function clearScreen() {
		context.fillStyle = bgColor;
		context.fillRect(0,0,displayWidth,displayHeight);
		orbitLayerContext.clearRect(0,0,displayWidth+1,displayHeight+1);
	}
	
	function clearParticleLayer() {
		particleLayerContext.clearRect(0,0,displayWidth+1,displayHeight+1);
	}
	
	function makeParticles(colors) {
		var i;
		
		particles = [];
		
		for (i = 0; i<numParticles; i++) {
			var phase = Math.PI*2*i/numParticles;
			var color;
			var trailColor;
			if (i<colors.length) {
				color = colorLookup[colors[i]];
				trailColor = trailColorLookup[colors[i]];
			}
			else {
				color = defaultParticleColor;
				trailColor = defaultTrailColor;
			}
					
			var p = {
					x: 0,
					y: 0,
					lastX: 0,
					lastY: 0,
					phase: phase,
					color: color,
					trailColor: trailColor
			}
			particles.push(p);
		}
		
		setParticlePositions(time);
		resetLastPositions();
	}
	
	function resetLastPositions() {
		//set initial last positions
		for (i = 0; i<numParticles; i++) {
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
		for (i = 0; i<numParticles; i++) {
			j = (i + 1) % numParticles;
			pixX = xPixRate*(particles[j].x - xMin);
			pixY = yPixRate*(particles[j].y - yMax);
			endPixX.push(pixX);
			endPixY.push(pixY);
			staticOrbitDrawPointsX.push(xPixRate*(particles[i].x - xMin));
			staticOrbitDrawPointsY.push(yPixRate*(particles[i].y - yMax));
		}
	}
	
	function drawLastSegments() {
		var i;
		var p;
		var pixX;
		var pixY;
		orbitLayerContext.strokeStyle = staticOrbitColor;
		orbitLayerContext.lineWidth = staticOrbitWidth;

		for (i = 0; i < numParticles; i++) {
			p = particles[i];
			pixX = xPixRate*(p.x - xMin);
			pixY = yPixRate*(p.y - yMax);
			
			orbitLayerContext.beginPath();
			orbitLayerContext.moveTo(endPixX[i],endPixY[i]);
			orbitLayerContext.lineTo(staticOrbitDrawPointsX[i], staticOrbitDrawPointsY[i]);
			orbitLayerContext.stroke();
		}
	}
	
	function drawParticles() {
		var i;
		var len;
		var pixX;
		var pixY;
		var lastPixX;
		var lastPixY;
		var p;
		var dx;
		var dy;
		
		context.lineCap = "round";
		
		len = particles.length;
		for (i = 0; i < len; i++) {
			p = particles[i];
			pixX = xPixRate*(p.x - xMin);
			pixY = yPixRate*(p.y - yMax);
			lastPixX = xPixRate*(p.lastX - xMin);
			lastPixY = yPixRate*(p.lastY - yMax);
						
			//particle
			particleLayerContext.strokeStyle = "rgba(0,0,0,0.5)"
			particleLayerContext.lineWidth = 2;
			particleLayerContext.fillStyle = p.color;
			particleLayerContext.beginPath();
			particleLayerContext.arc(pixX,pixY,particleRad+1,0,Math.PI*2,false);
			particleLayerContext.closePath();
			particleLayerContext.fill();
			particleLayerContext.stroke();
			
			if (trajectoriesOn) {
				//trail
				context.strokeStyle = p.trailColor;
				context.lineWidth = trailWidth;
				context.beginPath();
				context.moveTo(lastPixX,lastPixY);
				context.lineTo(pixX, pixY);
				context.stroke();
				
				if (drawingStaticOrbit) {
					orbitLayerContext.strokeStyle = staticOrbitColor;
					orbitLayerContext.lineWidth = staticOrbitWidth;
					
					dx = staticOrbitDrawPointsX[i] - pixX;
					dy = staticOrbitDrawPointsY[i] - pixY;
					if (dx*dx + dy*dy > staticOrbitMinDrawDistance*staticOrbitMinDrawDistance) {				
						orbitLayerContext.beginPath();
						orbitLayerContext.moveTo(staticOrbitDrawPointsX[i],staticOrbitDrawPointsY[i]);
						orbitLayerContext.lineTo(pixX, pixY);
						orbitLayerContext.stroke();
						staticOrbitDrawPointsX[i] = pixX;
						staticOrbitDrawPointsY[i] = pixY;
					}
				}
			}
		}

	}
	
	function setParticlePositions(t) {
		var i;
		var len;
		len = particles.length;
		for (i = 0; i < len; i++) {
			particles[i].lastX = particles[i].x;
			particles[i].lastY = particles[i].y;
			particles[i].x = fourierSum(t,xSinFreq, xSinCoeff, xCosFreq, xCosCoeff, particles[i].phase);
			particles[i].y = fourierSum(t,ySinFreq, ySinCoeff, yCosFreq, yCosCoeff, particles[i].phase);
		}
	}
	
	function fourierSum(t,sinFreqs,sinCoeffs,cosFreqs,cosCoeffs,phaseShift) {
		var i, len;
		var sum = 0;
		len = sinCoeffs.length;
		for (i = 0; i < len; i++) {
			sum += sinCoeffs[i]*Math.sin(sinFreqs[i]*(t + phaseShift));
		}
		len = cosCoeffs.length;
		for (i = 0; i < len; i++) {
			sum += cosCoeffs[i]*Math.cos(cosFreqs[i]*(t + phaseShift));
		}
		return sum;
	}
		
	function speedSliderHandler() {
		setTInc();	
	}
	
	function setTInc() {
		tInc = timeFactor*(tIncMin + (tIncMax - tIncMin)*$("#speedSlider").slider("value"));
	}
	
	function setOrbit(orbitIndex) {
		
		var orbitObject = jsonData.orbits[orbitIndex];
		
		setPlotWindow(orbitObject.plotWindow);
		
		// setInfoText(orbitObject.info);
		
		numParticles = orbitObject.numParticles;
		
		xSinFreq = [];
		xCosFreq = [];
		ySinFreq = [];
		yCosFreq = [];
		xSinCoeff = [];
		xCosCoeff = [];
		ySinCoeff = [];
		yCosCoeff = [];
		
		var arrays;
		
		arrays = separateArray(orbitObject.x.sin);
		xSinFreq = arrays.even.slice(0);
		xSinCoeff = arrays.odd.slice(0);
		
		arrays = separateArray(orbitObject.x.cos);
		xCosFreq = arrays.even.slice(0);
		xCosCoeff = arrays.odd.slice(0);
		
		arrays = separateArray(orbitObject.y.sin);
		ySinFreq = arrays.even.slice(0);
		ySinCoeff = arrays.odd.slice(0);
		
		arrays = separateArray(orbitObject.y.cos);
		yCosFreq = arrays.even.slice(0);
		yCosCoeff = arrays.odd.slice(0);
		
		var colors;
		if (!orbitObject.colors) {
			if (numParticles < colorLookup.length) {
				//if fewer than color list, default will be to do different colors.
				colors = [];
				for (var i = 0; i < numParticles; i++) {
					colors.push(i);
				}
			}
			else {
				//if more than color list, set to empty array,then default will be to make all same color.
				colors = [];
			}
		}
		else {
			colors = orbitObject.colors;
		}
		makeParticles(colors);
		clearScreen();
				
		if (!orbitObject.length) {
			timeFactor = 1;	
		}
		else {
			timeFactor = (orbitObject.plotWindow.xMax - orbitObject.plotWindow.xMin)/orbitObject.length;
		}
		
		time = 0;
		
		if (trajectoriesOn) {
			drawingStaticOrbit = true;
			orbitDrawStartTime = orbitDrawTime = time;
		}
		else {
			drawingStaticOrbit = false;
		}
		
		setParticlePositions(0);
		resetLastPositions();
		setStartPositions();
		
		//if stopped, draw particles in correct place
		if (!running) {
			clearParticleLayer();
			drawParticles();	
		}
		setTInc();
		
	}
	
	//function to split arrays
	function separateArray(array) {
		var returnObj = {even: [], odd: []};
		var i;
		var len = array.length;
		for (i = 0; i < len; i = i + 2) {
			returnObj.even.push(array[i]);
			returnObj.odd.push(array[i+1]);
		}
		return returnObj;
	}	
	
	function setData(dataObject) {
		jsonData = dataObject;
		numOrbits = jsonData.orbits.length;
		populateOrbitRadioButtons(dataObject);
	}
	
}
