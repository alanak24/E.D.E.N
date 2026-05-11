//
//  TVDetailView.swift
//  EDEN
//
//  Created by Alana Kumar on 1/5/2026.
//

import SwiftUI

struct TVDetailView: View {
    @State private var rating: Double = 0
    @EnvironmentObject var showVM: TVViewModel


    let show: TV
    
    var ratingBinding: Binding<Double> {
        Binding(
            get: {
                showVM.preferences[show.id]?.rating ?? 0
            },
            set: { newValue in
                showVM.setRating(newValue, for: show.id)
            }
        )
    }

    var isLiked: Bool {
        showVM.preferences[show.id]?.isLiked ?? false
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {

                AsyncImage(
                    url: URL(string:
                        "https://image.tmdb.org/t/p/w500\(show.poster_path)"
                    )
                ) { image in
                    image.resizable().scaledToFit()
                } placeholder: {
                    ProgressView()
                }

                Text(show.name)
                    .font(.title)
                    .foregroundColor(.white)
                
                Text(show.overview)
                    .foregroundColor(.gray)
                
                VStack(alignment: .leading, spacing: 12) {
                    
                    Text("Your Rating")
                        .font(.headline)
                    
                    StarRatingView(rating: ratingBinding)
                    
                    Button(action: {
                        showVM.toggleLike(for: show.id)
                    }) {
                        HStack {
                            Image(systemName: isLiked ? "heart.fill" : "heart")
                            Text(isLiked ? "Liked" : "Add to Likes")
                        }
                        .foregroundColor(.red)
                    }
                }


                // Optional: add overview later if you include it in model

            }
            .padding()
        }
        .background(Color.black)
        .navigationTitle("TV Details")
    }
}
