window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox = $('#imagebox')
		input = $('#imageinput')[0]
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('image' , input.files[0]);
			$.ajax({
				url: "http://localhost:5000/editImage", // fix this to your liking
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				except: "100-continue",
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					// alert("hello"); // if it's failing on actual server check your server FIREWALL + SET UP CORS
					bytestring = data['status']
					image = bytestring.split('\'')[1]
					returnimagebox = $('#returnimagebox')
					returnimagebox.attr('src' , 'data:image/jpeg;base64,'+image)
					returnimagebox.height('340px');
			        returnimagebox.width(window.width/3);

			        // plot histograms
			        hist_orig = data['hist_orig']
			        hist_edit = data['hist_edit']

                    toHist(hist_orig[0], hist_edit[0], 'black', 'hist')
			        toHist(hist_orig[1], hist_edit[1], 'red', 'rHist')
                    toHist(hist_orig[2], hist_edit[2], 'green', 'gHist')
                    toHist(hist_orig[3], hist_edit[3], 'blue', 'bHist')
				}
			});
		}
	});
};

function toHist(data1, data2, color, spot){
    var original = {
      x: data1,
      name: 'Original',
      type: "histogram",
      opacity: 0.25,
      marker: {
         color: color,
      },
      xbins: {
        end: 256,
        size: 1,
        start: 0

      },
        xaxis: {range: [0, 256]},
    };
    var edited = {
      x: data2,
      name: "Edited",
      type: "histogram",
      opacity: 0.75,
      marker: {
         color: color,
      },
      xbins: {
        end: 256,
        size: 1,
        start: 0

      },
        xaxis: {range: [0, 256]},
    };

    var data = [original, edited];
    var layout = {barmode: "overlay",
                  height: 170,
                  //width: 400,
                  margin: {
                        l: 10,
                        r: 30,
                        b: 30,
                        t: 30,
                        pad: 1
                  },
                  yaxis: {
                  showticklabels: false
                  }
  };
    Plotly.newPlot(spot, data, layout, {staticPlot: true, displayModeBar: true});
}

function readUrl(input){
	imagebox = $('#imagebox')
	console.log("evoked readUrl")
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){
			// console.log(e)

			imagebox.attr('src',e.target.result);
			imagebox.height('340px');
			imagebox.width(window.width/3);
		}
		reader.readAsDataURL(input.files[0]);
	}
}